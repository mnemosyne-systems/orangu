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

package ai.mnemosyne_systems.orangu.core

/**
 * Removes terminal escape sequences from streamed text. orangu's own output
 * is plain when it is not on a terminal, but a command it runs (git, for
 * one) may still colour its output. The text arrives in pieces, so a
 * sequence cut between two pieces is held back until the rest of it comes.
 */
class AnsiFilter {
    private enum class State { TEXT, ESCAPE, CSI, OSC, OSC_ESCAPE }

    private var state = State.TEXT

    /** The next piece of text, without its escape sequences. */
    fun push(piece: CharSequence): String {
        val text = StringBuilder(piece.length)
        for (c in piece) {
            when (state) {
                State.TEXT ->
                    if (c == ESCAPE) {
                        state = State.ESCAPE
                    } else if (c != '\r') {
                        text.append(c)
                    }
                // ESC [ starts a control sequence, ESC ] an operating system
                // command (a title, a link); anything else is a two-character
                // sequence.
                State.ESCAPE -> state = when (c) {
                    '[' -> State.CSI
                    ']' -> State.OSC
                    else -> State.TEXT
                }
                // Parameters and intermediates run up to a final byte between
                // '@' and '~'.
                State.CSI -> if (c in '@'..'~') state = State.TEXT
                State.OSC -> when (c) {
                    BELL -> state = State.TEXT
                    ESCAPE -> state = State.OSC_ESCAPE
                }
                // ESC \ ends it; a stray ESC inside it is dropped.
                State.OSC_ESCAPE -> state = if (c == '\\') State.TEXT else State.OSC
            }
        }
        return text.toString()
    }

    private companion object {
        const val ESCAPE = '\u001b'
        const val BELL = '\u0007'
    }
}

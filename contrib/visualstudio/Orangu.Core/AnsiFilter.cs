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

using System.Text;

namespace Orangu.Core;

/// <summary>
/// Removes terminal escape sequences from streamed text. orangu's own output
/// is plain when it is not on a terminal, but a command it runs (git, for
/// one) may still colour its output. The text arrives in pieces, so a
/// sequence cut between two pieces is held back until the rest of it comes.
/// </summary>
public sealed class AnsiFilter
{
    private const char Escape = '\u001b';
    private const char Bell = '\u0007';

    private State state = State.Text;

    private enum State
    {
        Text,
        Escape,
        Csi,
        Osc,
        OscEscape,
    }

    /// <summary>The next piece of text, without its escape sequences.</summary>
    public string Push(string piece)
    {
        var text = new StringBuilder(piece.Length);
        foreach (char c in piece)
        {
            switch (this.state)
            {
                case State.Text:
                    if (c == Escape)
                    {
                        this.state = State.Escape;
                    }
                    else if (c != '\r')
                    {
                        text.Append(c);
                    }

                    break;
                case State.Escape:
                    // ESC [ starts a control sequence, ESC ] an operating
                    // system command (a title, a link); anything else is a
                    // two-character sequence.
                    this.state = c switch
                    {
                        '[' => State.Csi,
                        ']' => State.Osc,
                        _ => State.Text,
                    };
                    break;
                case State.Csi:
                    // Parameters and intermediates run up to a final byte
                    // between '@' and '~'.
                    if (c >= '@' && c <= '~')
                    {
                        this.state = State.Text;
                    }

                    break;
                case State.Osc:
                    if (c == Bell)
                    {
                        this.state = State.Text;
                    }
                    else if (c == Escape)
                    {
                        this.state = State.OscEscape;
                    }

                    break;
                case State.OscEscape:
                    // ESC \ ends it; a stray ESC inside it is dropped.
                    this.state = c == '\\' ? State.Text : State.Osc;
                    break;
            }
        }

        return text.ToString();
    }
}

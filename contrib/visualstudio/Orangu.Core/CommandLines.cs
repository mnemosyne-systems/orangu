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

using System.Diagnostics;
using System.Text;

namespace Orangu.Core;

/// <summary>The two ways the extension starts orangu.</summary>
public static class CommandLines
{
    /// <summary>
    /// <c>orangu -w WORKSPACE -p PROMPT</c>: one prompt, the answer streamed on
    /// stdout and the notes about what it is doing on stderr. Nothing is
    /// asked on stdin, so it runs without a console.
    /// </summary>
    public static ProcessStartInfo OneShot(string binary, string workspace, string prompt)
    {
        var info = new ProcessStartInfo(binary)
        {
            WorkingDirectory = workspace,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            StandardOutputEncoding = new UTF8Encoding(false),
            StandardErrorEncoding = new UTF8Encoding(false),
        };
        info.ArgumentList.Add("-w");
        info.ArgumentList.Add(workspace);
        info.ArgumentList.Add("-p");
        info.ArgumentList.Add(prompt);

        // Plain text: the tool window cannot show colours.
        info.Environment["NO_COLOR"] = "1";
        return info;
    }

    /// <summary>
    /// <c>orangu -w WORKSPACE</c> in a console window of its own: the full
    /// interface, with /review, /auto_review and the session kept.
    /// </summary>
    public static ProcessStartInfo Interactive(string binary, string workspace)
    {
        // Through the shell, so it gets a new console (or Windows Terminal)
        // instead of the extension host's hidden one. The shell takes one
        // argument string, not a list.
        return new ProcessStartInfo(binary)
        {
            WorkingDirectory = workspace,
            UseShellExecute = true,
            Arguments = "-w " + QuoteWindowsArgument(workspace),
        };
    }

    /// <summary>
    /// Quotes one argument the way the Windows C runtime reads it back:
    /// backslashes are literal unless they come before a quote.
    /// </summary>
    public static string QuoteWindowsArgument(string argument)
    {
        if (argument.Length > 0 && argument.IndexOfAny(new[] { ' ', '\t', '\n', '\v', '"' }) < 0)
        {
            return argument;
        }

        var quoted = new StringBuilder("\"");
        int backslashes = 0;
        foreach (char c in argument)
        {
            if (c == '\\')
            {
                backslashes++;
                continue;
            }

            if (c == '"')
            {
                // Each backslash before a quote is doubled, and the quote escaped.
                quoted.Append('\\', (backslashes * 2) + 1);
            }
            else
            {
                quoted.Append('\\', backslashes);
            }

            backslashes = 0;
            quoted.Append(c);
        }

        // Before the closing quote, backslashes are doubled so it stays a quote.
        quoted.Append('\\', backslashes * 2);
        quoted.Append('"');
        return quoted.ToString();
    }
}

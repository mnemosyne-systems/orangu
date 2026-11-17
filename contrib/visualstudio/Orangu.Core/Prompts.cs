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

using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace Orangu.Core;

/// <summary>A selection in the editor, attached to the next prompt.</summary>
/// <param name="FilePath">The document's full path.</param>
/// <param name="StartLine">The first selected line, from 1.</param>
/// <param name="EndLine">The last selected line, from 1.</param>
/// <param name="Text">The selected text.</param>
public sealed record EditorSelection(string FilePath, int StartLine, int EndLine, string Text);

/// <summary>The prompts the editor commands send.</summary>
public static class Prompts
{
    /// <summary>What the prompt box is filled with after a selection is attached.</summary>
    public const string DefaultSelectionQuestion = "Explain what this code does.";

    private static readonly Dictionary<string, string> Languages = new(StringComparer.OrdinalIgnoreCase)
    {
        [".c"] = "c",
        [".h"] = "c",
        [".cc"] = "cpp",
        [".cpp"] = "cpp",
        [".cxx"] = "cpp",
        [".hpp"] = "cpp",
        [".cs"] = "csharp",
        [".fs"] = "fsharp",
        [".vb"] = "vb",
        [".rs"] = "rust",
        [".py"] = "python",
        [".js"] = "javascript",
        [".ts"] = "typescript",
        [".java"] = "java",
        [".go"] = "go",
        [".sql"] = "sql",
        [".json"] = "json",
        [".xml"] = "xml",
        [".xaml"] = "xml",
        [".csproj"] = "xml",
        [".yml"] = "yaml",
        [".yaml"] = "yaml",
        [".md"] = "markdown",
        [".sh"] = "bash",
        [".ps1"] = "powershell",
        [".toml"] = "toml",
    };

    /// <summary>
    /// A question about a selection: the question, then where the code is and
    /// the code itself, so the model can also open the file around it.
    /// </summary>
    public static string AboutSelection(string workspace, EditorSelection selection, string question)
    {
        question = string.IsNullOrWhiteSpace(question) ? DefaultSelectionQuestion : question.Trim();
        string lines = selection.StartLine == selection.EndLine
            ? $"line {selection.StartLine}"
            : $"lines {selection.StartLine}-{selection.EndLine}";
        string code = selection.Text.TrimEnd('\r', '\n').Replace("\r\n", "\n");
        string fence = Fence(code);

        return new StringBuilder()
            .Append(question)
            .Append("\n\nThis is ")
            .Append(lines)
            .Append(" of ")
            .Append(DisplayPath(workspace, selection.FilePath))
            .Append(":\n\n")
            .Append(fence)
            .Append(Language(selection.FilePath))
            .Append('\n')
            .Append(code)
            .Append('\n')
            .Append(fence)
            .ToString();
    }

    /// <summary>
    /// A review of one file. /review needs orangu's terminal interface, so the
    /// tool window asks for the review in words; the model reads the file
    /// with its own tools.
    /// </summary>
    public static string ReviewFile(string workspace, string filePath)
    {
        return $"Review {DisplayPath(workspace, filePath)}. Read the whole file and point out bugs, "
            + "risky code, missing error handling and missing tests, each with its line number "
            + "and a suggested fix. Do not change any files.";
    }

    /// <summary>
    /// The short form shown for an attached selection: <c>src/a.cs:3-9</c>.
    /// </summary>
    public static string SelectionLabel(string workspace, EditorSelection selection)
    {
        string path = DisplayPath(workspace, selection.FilePath);
        return selection.StartLine == selection.EndLine
            ? $"{path}:{selection.StartLine}"
            : $"{path}:{selection.StartLine}-{selection.EndLine}";
    }

    /// <summary>
    /// A file as orangu's tools take it: relative to the workspace with
    /// forward slashes, or the full path when it is outside it.
    /// </summary>
    public static string DisplayPath(string workspace, string filePath)
    {
        string relative = Path.GetRelativePath(workspace, filePath);
        bool outside = relative == ".."
            || relative.StartsWith(".." + Path.DirectorySeparatorChar, StringComparison.Ordinal)
            || relative.StartsWith("../", StringComparison.Ordinal)
            || Path.IsPathRooted(relative);
        return outside ? filePath : relative.Replace('\\', '/');
    }

    private static string Language(string filePath)
    {
        return Languages.TryGetValue(Path.GetExtension(filePath), out string? language) ? language : string.Empty;
    }

    /// <summary>A fence longer than any run of backticks in the code.</summary>
    private static string Fence(string code)
    {
        int longest = 0;
        int run = 0;
        foreach (char c in code)
        {
            run = c == '`' ? run + 1 : 0;
            longest = Math.Max(longest, run);
        }

        return new string('`', Math.Max(3, longest + 1));
    }
}

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

using System.IO;
using Orangu.Core;
using Xunit;

namespace Orangu.Core.Tests;

public class PromptsTests
{
    private static readonly string Workspace = Path.Combine(Path.GetTempPath(), "app");

    private static string InWorkspace(params string[] parts) => Path.Combine(Workspace, Path.Combine(parts));

    [Fact]
    public void A_question_about_a_selection_carries_the_place_and_the_code()
    {
        var selection = new EditorSelection(InWorkspace("src", "Parser.cs"), 12, 14, "int a = 1;\r\nint b = 2;\r\nreturn a + b;\r\n");

        string prompt = Prompts.AboutSelection(Workspace, selection, "  Why is b needed? ");

        Assert.Equal(
            "Why is b needed?\n\nThis is lines 12-14 of src/Parser.cs:\n\n```csharp\nint a = 1;\nint b = 2;\nreturn a + b;\n```",
            prompt);
    }

    [Fact]
    public void One_line_and_no_question_ask_for_an_explanation()
    {
        var selection = new EditorSelection(InWorkspace("main.rs"), 3, 3, "let x = y?;");

        string prompt = Prompts.AboutSelection(Workspace, selection, " ");

        Assert.StartsWith(Prompts.DefaultSelectionQuestion + "\n\nThis is line 3 of main.rs:\n\n```rust\n", prompt);
    }

    [Fact]
    public void Code_holding_a_fence_gets_a_longer_one()
    {
        var selection = new EditorSelection(InWorkspace("README.notes"), 1, 3, "```\ncode\n```");

        string prompt = Prompts.AboutSelection(Workspace, selection, "Fix it");

        Assert.EndsWith(":\n\n````\n```\ncode\n```\n````", prompt);
    }

    [Fact]
    public void A_review_names_the_file_and_asks_for_no_edits()
    {
        string prompt = Prompts.ReviewFile(Workspace, InWorkspace("src", "Db.cs"));

        Assert.StartsWith("Review src/Db.cs.", prompt);
        Assert.Contains("line number", prompt);
        Assert.EndsWith("Do not change any files.", prompt);
    }

    [Fact]
    public void A_file_outside_the_workspace_keeps_its_full_path()
    {
        string outside = Path.Combine(Path.GetTempPath(), "elsewhere", "x.cs");

        Assert.Equal(outside, Prompts.DisplayPath(Workspace, outside));
    }

    [Fact]
    public void The_label_is_the_path_and_the_lines()
    {
        Assert.Equal("src/A.cs:3-9", Prompts.SelectionLabel(Workspace, new EditorSelection(InWorkspace("src", "A.cs"), 3, 9, "x")));
        Assert.Equal("A.cs:4", Prompts.SelectionLabel(Workspace, new EditorSelection(InWorkspace("A.cs"), 4, 4, "x")));
    }
}

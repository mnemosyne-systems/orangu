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

using System.Collections.Generic;
using System.IO;
using Orangu.Core;
using Xunit;

namespace Orangu.Core.Tests;

public class WorkspaceChooserTests
{
    private static readonly string Root = Path.Combine(Path.GetTempPath(), "repo");

    [Fact]
    public void The_solution_folder_wins()
    {
        string solution = Path.Combine(Root, "sln");
        var existing = new HashSet<string> { solution, Path.Combine(Root, ".git") };

        Assert.Equal(solution, WorkspaceChooser.Choose(solution, Path.Combine(Root, "a.cs"), existing.Contains));
    }

    [Fact]
    public void A_lone_file_runs_in_its_git_repository()
    {
        var existing = new HashSet<string> { Path.Combine(Root, ".git") };

        string? workspace = WorkspaceChooser.Choose(null, Path.Combine(Root, "src", "deep", "a.cs"), existing.Contains);

        Assert.Equal(Root, workspace);
    }

    [Fact]
    public void A_lone_file_outside_git_runs_in_its_folder()
    {
        string file = Path.Combine(Root, "notes", "a.cs");

        Assert.Equal(Path.Combine(Root, "notes"), WorkspaceChooser.Choose("", file, _ => false));
    }

    [Fact]
    public void A_solution_folder_that_is_gone_falls_back_to_the_document()
    {
        string file = Path.Combine(Root, "a.cs");

        Assert.Equal(Root, WorkspaceChooser.Choose(Path.Combine(Root, "gone"), file, _ => false));
    }

    [Fact]
    public void Nothing_open_is_no_workspace()
    {
        Assert.Null(WorkspaceChooser.Choose(null, null, _ => true));
    }
}

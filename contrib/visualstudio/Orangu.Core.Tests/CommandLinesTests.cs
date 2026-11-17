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
using Orangu.Core;
using Xunit;

namespace Orangu.Core.Tests;

public class CommandLinesTests
{
    [Fact]
    public void A_one_shot_passes_the_workspace_and_the_prompt_as_separate_arguments()
    {
        ProcessStartInfo info = CommandLines.OneShot("/bin/orangu", "/src/my app", "what does \"main\" do?\n/status");

        Assert.Equal("/bin/orangu", info.FileName);
        Assert.Equal("/src/my app", info.WorkingDirectory);
        Assert.Equal(new[] { "-w", "/src/my app", "-p", "what does \"main\" do?\n/status" }, info.ArgumentList);
        Assert.Empty(info.Arguments);
    }

    [Fact]
    public void A_one_shot_runs_hidden_with_its_output_captured()
    {
        ProcessStartInfo info = CommandLines.OneShot("orangu", "/src", "hi");

        Assert.False(info.UseShellExecute);
        Assert.True(info.CreateNoWindow);
        Assert.True(info.RedirectStandardInput);
        Assert.True(info.RedirectStandardOutput);
        Assert.True(info.RedirectStandardError);
        Assert.Equal("1", info.Environment["NO_COLOR"]);
    }

    [Fact]
    public void The_interactive_one_gets_its_own_window_in_the_workspace()
    {
        ProcessStartInfo info = CommandLines.Interactive(@"C:\bin\orangu.exe", @"C:\My Code\app\");

        Assert.True(info.UseShellExecute);
        Assert.Equal(@"C:\My Code\app\", info.WorkingDirectory);
        Assert.Equal(@"-w ""C:\My Code\app\\""", info.Arguments);
    }

    [Theory]
    [InlineData(@"C:\src\app", @"C:\src\app")]
    [InlineData(@"C:\My Code", @"""C:\My Code""")]
    [InlineData(@"C:\My Code\", @"""C:\My Code\\""")]
    [InlineData(@"say ""hi""", @"""say \""hi\""""")]
    [InlineData(@"a\""b c", @"""a\\\""b c""")]
    [InlineData("", @"""""")]
    public void Windows_arguments_are_quoted_so_they_read_back_the_same(string argument, string expected)
    {
        Assert.Equal(expected, CommandLines.QuoteWindowsArgument(argument));
    }
}

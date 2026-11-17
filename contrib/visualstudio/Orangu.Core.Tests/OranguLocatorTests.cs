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

public class OranguLocatorTests
{
    [Fact]
    public void A_configured_binary_is_used_as_it_is()
    {
        var files = new HashSet<string> { "/opt/orangu/orangu" };

        LocateResult result = OranguLocator.Locate(" \"/opt/orangu/orangu\" ", "/usr/bin", "/home/me", false, files.Contains);

        Assert.Equal("/opt/orangu/orangu", result.Path);
        Assert.Null(result.Error);
    }

    [Fact]
    public void A_configured_folder_means_the_binary_inside_it()
    {
        string expected = Path.Combine("/opt/orangu", "orangu");
        var files = new HashSet<string> { expected };

        LocateResult result = OranguLocator.Locate("/opt/orangu", null, null, false, files.Contains);

        Assert.Equal(expected, result.Path);
    }

    [Fact]
    public void A_configured_path_that_is_not_there_is_not_replaced_by_a_search()
    {
        // orangu is on the PATH, but the setting names another one: using the
        // PATH's would quietly run a different binary than the one asked for.
        var files = new HashSet<string> { Path.Combine("/usr/bin", "orangu") };

        LocateResult result = OranguLocator.Locate("/opt/gone/orangu", "/usr/bin", null, false, files.Contains);

        Assert.Null(result.Path);
        Assert.Contains("/opt/gone/orangu", result.Error);
    }

    [Fact]
    public void Windows_searches_the_path_for_the_exe_in_order()
    {
        string first = Path.Combine(@"C:\tools", "orangu.exe");
        string second = Path.Combine(@"C:\other", "orangu.exe");
        var files = new HashSet<string> { first, second, Path.Combine(@"C:\bin", "orangu") };

        LocateResult result = OranguLocator.Locate(null, @"C:\bin;;""C:\tools"";C:\other", null, true, files.Contains);

        Assert.Equal(first, result.Path);
    }

    [Fact]
    public void Unix_splits_the_path_on_colons()
    {
        string expected = Path.Combine("/usr/local/bin", "orangu");
        var files = new HashSet<string> { expected };

        LocateResult result = OranguLocator.Locate("", "/usr/bin:/usr/local/bin", null, false, files.Contains);

        Assert.Equal(expected, result.Path);
    }

    [Fact]
    public void Cargo_bin_is_looked_in_when_the_path_does_not_have_it()
    {
        string expected = Path.Combine("/home/me", ".cargo", "bin", "orangu");
        var files = new HashSet<string> { expected };

        LocateResult result = OranguLocator.Locate(null, "/usr/bin", "/home/me", false, files.Contains);

        Assert.Equal(expected, result.Path);
    }

    [Fact]
    public void Nothing_found_says_how_to_fix_it()
    {
        LocateResult result = OranguLocator.Locate(null, @"C:\bin", @"C:\Users\me", true, _ => false);

        Assert.Null(result.Path);
        Assert.Contains("orangu.exe", result.Error);
        Assert.Contains("Tools > Options > orangu", result.Error);
    }
}

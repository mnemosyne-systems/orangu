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

using Orangu.Core;
using Xunit;

namespace Orangu.Core.Tests;

public class AnsiFilterTests
{
    [Fact]
    public void Plain_text_passes_through()
    {
        Assert.Equal("fn main() {}\n- a `b`\n", new AnsiFilter().Push("fn main() {}\n- a `b`\n"));
    }

    [Fact]
    public void Colours_are_removed()
    {
        var filter = new AnsiFilter();

        Assert.Equal("* abc1234 (HEAD) fix it", filter.Push("* \u001b[33mabc1234\u001b[m (\u001b[1;36mHEAD\u001b[m) fix it"));
    }

    [Fact]
    public void A_sequence_split_between_pieces_is_still_removed()
    {
        var filter = new AnsiFilter();

        string text = filter.Push("red: \u001b") + filter.Push("[3") + filter.Push("1mbad\u001b[") + filter.Push("0m done");

        Assert.Equal("red: bad done", text);
    }

    [Fact]
    public void Titles_and_links_are_removed_with_either_ending()
    {
        var filter = new AnsiFilter();

        string text = filter.Push("\u001b]0;orangu\u0007a \u001b]8;;https://example.org\u001b\\link\u001b]8;;\u001b\\ b");

        Assert.Equal("a link b", text);
    }

    [Fact]
    public void Carriage_returns_are_dropped()
    {
        Assert.Equal("one\ntwo\n", new AnsiFilter().Push("one\r\ntwo\r\n"));
    }
}

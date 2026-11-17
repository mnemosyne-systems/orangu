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
using System.ComponentModel;
using System.IO;
using System.Threading.Tasks;
using Orangu.Core;
using Orangu.VisualStudio.ToolWindow;
using Xunit;

namespace Orangu.VisualStudio.Tests;

public sealed class ToolWindowDataTests : System.IDisposable
{
    // Nothing here reaches Visual Studio: attaching, the transcript and the
    // buttons are the session's own state.
    private readonly OranguSession session = new(null!);

    private OranguToolWindowData Data => this.session.Data;

    public void Dispose() => this.session.Dispose();

    [Fact]
    public void Running_swaps_ask_for_stop()
    {
        Assert.True(this.Data.AskCommand.CanExecute);
        Assert.False(this.Data.StopCommand.CanExecute);

        this.Data.IsRunning = true;
        Assert.False(this.Data.AskCommand.CanExecute);
        Assert.True(this.Data.StopCommand.CanExecute);

        this.Data.IsRunning = false;
        Assert.True(this.Data.AskCommand.CanExecute);
        Assert.False(this.Data.StopCommand.CanExecute);
    }

    [Fact]
    public void While_running_the_answer_is_sent_in_batches_and_all_of_it_at_the_end()
    {
        var changes = new List<string>();
        this.Data.PropertyChanged += (_, e) => changes.Add(e.PropertyName!);

        this.Data.IsRunning = true;
        this.Data.Append("Hel");
        this.Data.Append("lo");
        Assert.Equal(string.Empty, this.Data.Transcript);

        this.Data.IsRunning = false;
        Assert.Equal("Hello", this.Data.Transcript);
        Assert.Single(changes, nameof(OranguToolWindowData.Transcript));
    }

    [Fact]
    public void When_idle_a_line_shows_at_once_and_clear_empties_it()
    {
        this.Data.Append("note\n");
        Assert.Equal("note\n", this.Data.Transcript);

        this.Data.ClearTranscript();
        Assert.Equal(string.Empty, this.Data.Transcript);
    }

    [Fact]
    public void A_long_transcript_keeps_its_latest_part()
    {
        this.Data.Append(new string('a', 150_000));
        this.Data.Append(new string('b', 100_000));

        Assert.Equal(200_000, this.Data.Transcript.Length);
        Assert.EndsWith(new string('b', 100_000), this.Data.Transcript);
    }

    [Fact]
    public void Attaching_a_selection_labels_it_and_suggests_a_question()
    {
        string workspace = Path.Combine(Path.GetTempPath(), "app");
        var changes = new List<string>();
        this.Data.PropertyChanged += (_, e) => changes.Add(e.PropertyName!);

        this.session.Attach(workspace, new EditorSelection(Path.Combine(workspace, "src", "A.cs"), 3, 9, "x"));

        Assert.Equal("src/A.cs:3-9", this.Data.Attachment);
        Assert.True(this.Data.HasAttachment);
        Assert.Contains(nameof(OranguToolWindowData.HasAttachment), changes);
        Assert.Equal(Prompts.DefaultSelectionQuestion, this.Data.Prompt);
    }

    [Fact]
    public async Task A_question_already_typed_is_kept_and_removing_the_selection_clears_it()
    {
        string workspace = Path.Combine(Path.GetTempPath(), "app");
        this.Data.Prompt = "Is this thread safe?";

        this.session.Attach(workspace, new EditorSelection(Path.Combine(workspace, "A.cs"), 1, 1, "x"));
        Assert.Equal("Is this thread safe?", this.Data.Prompt);

        await this.Data.DetachCommand.ExecuteAsync(null, null!, default);
        Assert.False(this.Data.HasAttachment);
        Assert.Equal(string.Empty, this.Data.Attachment);
    }

    [Fact]
    public async Task Stop_with_nothing_running_does_nothing()
    {
        await this.Data.StopCommand.ExecuteAsync(null, null!, default);

        Assert.False(this.Data.IsRunning);
    }

    [Fact]
    public void Changes_are_announced_to_the_window()
    {
        var changes = new List<string>();
        ((INotifyPropertyChanged)this.Data).PropertyChanged += (_, e) => changes.Add(e.PropertyName!);

        this.Data.Status = "Done.";
        this.Data.Prompt = "hi";
        this.Data.Status = "Done.";

        Assert.Equal(new[] { nameof(OranguToolWindowData.Status), nameof(OranguToolWindowData.Prompt) }, changes);
    }
}

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
using System.Collections.Concurrent;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Orangu.Core;
using Xunit;

namespace Orangu.Core.Tests;

/// <summary>
/// The runner against a stand-in for orangu: a shell script, or a batch file
/// on Windows, that answers the way <c>orangu -p</c> does.
/// </summary>
public sealed class OneShotRunnerTests : IDisposable
{
    private const string Esc = "\u001b";

    private readonly string folder = Directory.CreateTempSubdirectory("orangu-vs-").FullName;

    public void Dispose()
    {
        try
        {
            Directory.Delete(this.folder, recursive: true);
        }
        catch (IOException)
        {
            // A killed process may still hold it for a moment.
        }
    }

    [Fact]
    public async Task The_answer_and_the_notes_come_back_apart_and_without_colours()
    {
        string orangu = this.FakeOrangu(
            unix: $"printf 'Calling show_file\\n' >&2\nprintf '%s|' \"$@\"\nprintf '\\n{Esc}[32mok{Esc}[0m\\n'\n",
            windows: $"echo Calling show_file 1>&2\r\necho %1^|%~2^|%3^|%~4^|\r\necho {Esc}[32mok{Esc}[0m\r\n");
        var answer = new StringBuilder();
        var notes = new ConcurrentQueue<string>();

        OneShotResult result = await OneShotRunner.RunAsync(
            CommandLines.OneShot(orangu, this.folder, "what is here?"),
            text => answer.Append(text),
            notes.Enqueue,
            CancellationToken.None);

        Assert.Equal(OneShotOutcome.Succeeded, result.Outcome);
        Assert.Equal($"-w|{this.folder}|-p|what is here?|\nok\n", answer.ToString().Replace(" \n", "\n"));
        Assert.Equal(new[] { "Calling show_file" }, notes);
    }

    [Fact]
    public async Task A_failure_reports_the_last_thing_orangu_said()
    {
        string orangu = this.FakeOrangu(
            unix: "echo 'Connecting' >&2\necho 'Error: no server answers at localhost:8080' >&2\nexit 3\n",
            windows: "echo Connecting 1>&2\r\necho Error: no server answers at localhost:8080 1>&2\r\nexit /b 3\r\n");

        OneShotResult result = await OneShotRunner.RunAsync(
            CommandLines.OneShot(orangu, this.folder, "hi"), _ => { }, _ => { }, CancellationToken.None);

        Assert.Equal(OneShotOutcome.Failed, result.Outcome);
        Assert.Equal(3, result.ExitCode);
        Assert.Equal("Error: no server answers at localhost:8080", result.Error);
    }

    [Fact]
    public async Task Stopping_ends_orangu_and_what_it_started()
    {
        string orangu = this.FakeOrangu(
            unix: "echo started\nsleep 60\necho finished\n",
            windows: "echo started\r\nping -n 60 127.0.0.1 > nul\r\necho finished\r\n");
        using var stop = new CancellationTokenSource();
        var answer = new StringBuilder();
        var clock = Stopwatch.StartNew();

        Task<OneShotResult> run = OneShotRunner.RunAsync(
            CommandLines.OneShot(orangu, this.folder, "hi"),
            text =>
            {
                lock (answer)
                {
                    answer.Append(text);
                }

                stop.Cancel();
            },
            _ => { },
            stop.Token);
        OneShotResult result = await run.WaitAsync(TimeSpan.FromSeconds(30));

        Assert.Equal(OneShotOutcome.Cancelled, result.Outcome);
        Assert.True(clock.Elapsed < TimeSpan.FromSeconds(30));
        Assert.DoesNotContain("finished", answer.ToString());
    }

    [Fact]
    public async Task A_binary_that_is_not_there_does_not_start()
    {
        string missing = Path.Combine(this.folder, "no-such-orangu");

        OneShotResult result = await OneShotRunner.RunAsync(
            CommandLines.OneShot(missing, this.folder, "hi"), _ => { }, _ => { }, CancellationToken.None);

        Assert.Equal(OneShotOutcome.NotStarted, result.Outcome);
        Assert.Contains(missing, result.Error);
    }

    private string FakeOrangu(string unix, string windows)
    {
        if (OperatingSystem.IsWindows())
        {
            string batch = Path.Combine(this.folder, "orangu.cmd");
            File.WriteAllText(batch, "@echo off\r\n" + windows, new UTF8Encoding(false));
            return batch;
        }

        string script = Path.Combine(this.folder, "orangu");
        File.WriteAllText(script, "#!/bin/sh\n" + unix, new UTF8Encoding(false));
        File.SetUnixFileMode(script, UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute);
        return script;
    }
}

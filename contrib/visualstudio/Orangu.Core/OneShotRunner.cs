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
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

namespace Orangu.Core;

/// <summary>How a one-shot ended.</summary>
public enum OneShotOutcome
{
    /// <summary>orangu exited with 0.</summary>
    Succeeded,

    /// <summary>orangu exited with an error; its stderr says why.</summary>
    Failed,

    /// <summary>The user stopped it.</summary>
    Cancelled,

    /// <summary>orangu could not be started at all.</summary>
    NotStarted,
}

/// <summary>The end of a one-shot.</summary>
public sealed record OneShotResult(OneShotOutcome Outcome, int? ExitCode, string? Error);

/// <summary>
/// Runs a process built by <see cref="CommandLines.OneShot"/> and hands its
/// output over as it comes: the answer as it streams on stdout, and the notes
/// on stderr (the tools it calls, what it cost) line by line.
/// </summary>
public static class OneShotRunner
{
    /// <summary>Runs the one-shot to its end, or until it is cancelled.</summary>
    /// <param name="info">The process to start.</param>
    /// <param name="answer">Called with each piece of stdout, escapes removed.</param>
    /// <param name="note">Called with each line of stderr, escapes removed.</param>
    /// <param name="cancellation">Stops the run and every process it started.</param>
    public static async Task<OneShotResult> RunAsync(
        ProcessStartInfo info,
        Action<string> answer,
        Action<string> note,
        CancellationToken cancellation)
    {
        using var process = new Process { StartInfo = info };
        try
        {
            if (!process.Start())
            {
                return new OneShotResult(OneShotOutcome.NotStarted, null, $"{info.FileName} did not start");
            }
        }
        catch (Exception err) when (err is Win32Exception or InvalidOperationException or IOException)
        {
            return new OneShotResult(OneShotOutcome.NotStarted, null, $"{info.FileName}: {err.Message}");
        }

        if (info.RedirectStandardInput)
        {
            // Nothing is ever typed into a one-shot; a closed stdin makes
            // that plain to anything it runs.
            process.StandardInput.Close();
        }

        using var stop = cancellation.Register(() => Kill(process));
        string lastNote = string.Empty;
        Task stdout = PumpAsync(process.StandardOutput, new AnsiFilter(), answer);
        Task stderr = PumpLinesAsync(process.StandardError, line =>
        {
            lastNote = line;
            note(line);
        });

        await Task.WhenAll(stdout, stderr).ConfigureAwait(false);
        await process.WaitForExitAsync(CancellationToken.None).ConfigureAwait(false);

        if (cancellation.IsCancellationRequested)
        {
            return new OneShotResult(OneShotOutcome.Cancelled, null, null);
        }

        return process.ExitCode == 0
            ? new OneShotResult(OneShotOutcome.Succeeded, 0, null)
            : new OneShotResult(
                OneShotOutcome.Failed,
                process.ExitCode,
                lastNote.Length > 0 ? lastNote : $"orangu exited with {process.ExitCode}");
    }

    private static async Task PumpAsync(StreamReader reader, AnsiFilter filter, Action<string> sink)
    {
        // Read what is there, not whole lines: the answer streams a few
        // words at a time and should show up as it does.
        var buffer = new char[4096];
        int read;
        while ((read = await reader.ReadAsync(buffer.AsMemory()).ConfigureAwait(false)) > 0)
        {
            string text = filter.Push(new string(buffer, 0, read));
            if (text.Length > 0)
            {
                sink(text);
            }
        }
    }

    private static async Task PumpLinesAsync(StreamReader reader, Action<string> sink)
    {
        var filter = new AnsiFilter();
        string? line;
        while ((line = await reader.ReadLineAsync().ConfigureAwait(false)) is not null)
        {
            string text = filter.Push(line).TrimEnd();
            if (text.Length > 0)
            {
                sink(text);
            }
        }
    }

    private static void Kill(Process process)
    {
        try
        {
            process.Kill(entireProcessTree: true);
        }
        catch (InvalidOperationException)
        {
            // It had already exited.
        }
        catch (Win32Exception)
        {
            // It is exiting, or the tree changed while it was walked.
        }
    }
}

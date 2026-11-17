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
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.VisualStudio.Extensibility;
using Microsoft.VisualStudio.Extensibility.Editor;
using Microsoft.VisualStudio.Extensibility.Shell;
using Microsoft.VisualStudio.ProjectSystem.Query;
using Orangu.Core;
using Orangu.VisualStudio.ToolWindow;

namespace Orangu.VisualStudio;

/// <summary>
/// The tool window's state and the one-shots it runs. Each prompt is its own
/// <c>orangu -p</c> run in the workspace: one at a time, and no conversation
/// is kept between them.
/// </summary>
internal sealed class OranguSession : IDisposable
{
    private readonly VisualStudioExtensibility extensibility;
    private readonly object gate = new();
    private CancellationTokenSource? running;
    private EditorSelection? attached;
    private string? attachedWorkspace;

    public OranguSession(VisualStudioExtensibility extensibility)
    {
        this.extensibility = extensibility;
        this.Data = new OranguToolWindowData(this);
    }

    /// <summary>The tool window's data context.</summary>
    public OranguToolWindowData Data { get; }

    /// <summary>Ask: sends the prompt box, with the attached selection if there is one.</summary>
    public async Task AskFromWindowAsync(IClientContext context, CancellationToken cancellationToken)
    {
        string question = this.Data.Prompt.Trim();
        EditorSelection? selection = this.attached;
        if (question.Length == 0 && selection is null)
        {
            this.Data.Status = "Type a prompt first.";
            return;
        }

        string? workspace = selection is null
            ? await this.WorkspaceAsync(context, cancellationToken)
            : this.attachedWorkspace;
        if (workspace is null)
        {
            this.Data.Status = "Open a solution, a folder or a file first: orangu runs in its folder.";
            return;
        }

        string prompt = selection is null ? question : Prompts.AboutSelection(workspace, selection, question);
        if (await this.RunAsync(workspace, question.Length > 0 ? question : Prompts.DefaultSelectionQuestion, prompt, cancellationToken))
        {
            this.Data.Prompt = string.Empty;
            await this.DetachAsync();
        }
    }

    /// <summary>Attaches a selection to the next prompt and suggests a question.</summary>
    public void Attach(string workspace, EditorSelection selection)
    {
        this.attached = selection;
        this.attachedWorkspace = workspace;
        this.Data.Attachment = Prompts.SelectionLabel(workspace, selection);
        if (this.Data.Prompt.Trim().Length == 0)
        {
            this.Data.Prompt = Prompts.DefaultSelectionQuestion;
        }

        this.Data.Status = "The selection goes with the next prompt. Change the question, then press Ask.";
    }

    /// <summary>Removes the attached selection.</summary>
    public Task DetachAsync()
    {
        this.attached = null;
        this.attachedWorkspace = null;
        this.Data.Attachment = string.Empty;
        return Task.CompletedTask;
    }

    /// <summary>Asks for a review of one file.</summary>
    public Task ReviewAsync(string workspace, string filePath, CancellationToken cancellationToken)
    {
        string display = Prompts.DisplayPath(workspace, filePath);
        return this.RunAsync(workspace, $"Review {display}", Prompts.ReviewFile(workspace, filePath), cancellationToken);
    }

    /// <summary>Stop: ends the running one-shot and anything it started.</summary>
    public void Stop()
    {
        // Under the lock, so the run cannot dispose it between the check and
        // the call. Cancelling only kills processes, so it does not block.
        lock (this.gate)
        {
            this.running?.Cancel();
        }
    }

    /// <summary>Clear: empties the transcript.</summary>
    public Task ClearAsync()
    {
        this.Data.ClearTranscript();
        return Task.CompletedTask;
    }

    /// <summary>Open orangu: the full interface in its own window.</summary>
    public async Task OpenInteractiveAsync(IClientContext context, CancellationToken cancellationToken)
    {
        string? workspace = await this.WorkspaceAsync(context, cancellationToken);
        if (workspace is null)
        {
            await this.ShowErrorAsync("Open a solution, a folder or a file first: orangu runs in its folder.", cancellationToken);
            return;
        }

        LocateResult orangu = await this.LocateAsync(cancellationToken);
        if (orangu.Path is null)
        {
            await this.ShowErrorAsync(orangu.Error!, cancellationToken);
            return;
        }

        try
        {
            using Process? process = Process.Start(CommandLines.Interactive(orangu.Path, workspace));
        }
        catch (Exception err) when (err is Win32Exception or InvalidOperationException)
        {
            await this.ShowErrorAsync($"Could not start {orangu.Path}: {err.Message}", cancellationToken);
        }
    }

    /// <summary>
    /// The folder orangu runs in: the solution's, or for a lone file its Git
    /// repository or folder.
    /// </summary>
    public async Task<string?> WorkspaceAsync(IClientContext? context, CancellationToken cancellationToken)
    {
        string? solutionDirectory = null;
        try
        {
            IQueryResults<ISolutionSnapshot> solutions = await this.extensibility.Workspaces().QuerySolutionAsync(
                solution => solution.With(s => s.Directory),
                cancellationToken);
            solutionDirectory = solutions.FirstOrDefault()?.Directory;
        }
        catch (Exception err) when (err is not OperationCanceledException)
        {
            // No solution system to ask, as with Open Folder: the document decides.
        }

        string? document = null;
        if (context is not null)
        {
            using ITextViewSnapshot? view = await context.GetActiveTextViewAsync(cancellationToken);
            document = view?.FilePath;
        }

        return WorkspaceChooser.Choose(solutionDirectory, document, path => Directory.Exists(path) || File.Exists(path));
    }

    public void Dispose()
    {
        lock (this.gate)
        {
            this.running?.Cancel();
            this.running?.Dispose();
            this.running = null;
        }

        this.Data.Dispose();
    }

    /// <summary>Runs one prompt; false when it could not start.</summary>
    private async Task<bool> RunAsync(string workspace, string heading, string prompt, CancellationToken cancellationToken)
    {
        CancellationTokenSource stop;
        lock (this.gate)
        {
            if (this.running is not null)
            {
                this.Data.Status = "orangu is still answering; press Stop first.";
                return false;
            }

            stop = new CancellationTokenSource();
            this.running = stop;
        }

        try
        {
            LocateResult orangu = await this.LocateAsync(cancellationToken);
            if (orangu.Path is null)
            {
                this.Data.Status = orangu.Error!;
                return false;
            }

            this.Data.Append($"> {heading}\n\n");
            this.Data.Status = $"Running orangu in {workspace}";
            this.Data.IsRunning = true;

            OneShotResult result = await OneShotRunner.RunAsync(
                CommandLines.OneShot(orangu.Path, workspace, prompt),
                this.Data.Append,
                note => this.Data.Status = note,
                stop.Token);

            this.Data.Append("\n\n");
            this.Data.Status = result.Outcome switch
            {
                OneShotOutcome.Succeeded => "Done.",
                OneShotOutcome.Cancelled => "Stopped.",
                _ => result.Error ?? "orangu failed.",
            };

            return result.Outcome != OneShotOutcome.NotStarted;
        }
        finally
        {
            this.Data.IsRunning = false;
            lock (this.gate)
            {
                this.running = null;
            }

            stop.Dispose();
        }
    }

    private async Task<LocateResult> LocateAsync(CancellationToken cancellationToken)
    {
        string configured = string.Empty;
        try
        {
            // The settings API is still marked preview in the 17.14 SDK; see OranguSettings.
#pragma warning disable VSEXTPREVIEW_SETTINGS
            var value = await this.extensibility.Settings().ReadEffectiveValueAsync(OranguSettings.BinaryPath, cancellationToken);
            configured = value.ValueOrDefault(string.Empty);
#pragma warning restore VSEXTPREVIEW_SETTINGS
        }
        catch (Exception err) when (err is not OperationCanceledException)
        {
            // Unreadable settings: search, as an empty setting does.
        }

        return OranguLocator.Locate(configured);
    }

    private async Task ShowErrorAsync(string message, CancellationToken cancellationToken)
    {
        await this.extensibility.Shell().ShowPromptAsync(message, PromptOptions.OK, cancellationToken);
    }
}

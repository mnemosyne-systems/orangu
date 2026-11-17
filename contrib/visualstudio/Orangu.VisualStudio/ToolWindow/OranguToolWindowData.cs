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
using System.Runtime.Serialization;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.VisualStudio.Extensibility.UI;

namespace Orangu.VisualStudio.ToolWindow;

/// <summary>
/// What the tool window shows and its buttons. It lives in the extension's
/// process; Visual Studio draws it from the bindings in the XAML.
/// </summary>
[DataContract]
internal sealed class OranguToolWindowData : NotifyPropertyChangedObject, IDisposable
{
    // The transcript is sent whole on every change, so a long one is kept to
    // its latest part, and changes are sent a few times a second rather than
    // on every streamed word.
    private const int MaxTranscript = 200_000;
    private static readonly TimeSpan FlushInterval = TimeSpan.FromMilliseconds(120);

    private readonly object gate = new();
    private readonly StringBuilder transcript = new();
    private readonly Timer flush;
    private bool dirty;

    private string prompt = string.Empty;
    private string shownTranscript = string.Empty;
    private string status = "Type a prompt, or a command such as /status, and press Ask.";
    private string attachment = string.Empty;
    private bool isRunning;

    public OranguToolWindowData(OranguSession session)
    {
        this.flush = new Timer(_ => this.Flush(), null, Timeout.Infinite, Timeout.Infinite);
        this.AskCommand = new AsyncCommand((_, context, cancellationToken) => session.AskFromWindowAsync(context, cancellationToken));
        this.StopCommand = new AsyncCommand((_, _, _) =>
        {
            session.Stop();
            return Task.CompletedTask;
        })
        {
            CanExecute = false,
        };
        this.ClearCommand = new AsyncCommand((_, _, _) => session.ClearAsync());
        this.DetachCommand = new AsyncCommand((_, _, _) => session.DetachAsync());
        this.OpenCommand = new AsyncCommand((_, context, cancellationToken) => session.OpenInteractiveAsync(context, cancellationToken));
    }

    [DataMember]
    public string Prompt
    {
        get => this.prompt;
        set => this.SetProperty(ref this.prompt, value);
    }

    [DataMember]
    public string Transcript
    {
        get => this.shownTranscript;
        private set => this.SetProperty(ref this.shownTranscript, value);
    }

    [DataMember]
    public string Status
    {
        get => this.status;
        set => this.SetProperty(ref this.status, value);
    }

    /// <summary>The attached selection's label, such as src/a.cs:3-9; empty for none.</summary>
    [DataMember]
    public string Attachment
    {
        get => this.attachment;
        set
        {
            if (this.SetProperty(ref this.attachment, value))
            {
                this.RaiseNotifyPropertyChangedEvent(nameof(this.HasAttachment));
            }
        }
    }

    [DataMember]
    public bool HasAttachment => this.attachment.Length > 0;

    [DataMember]
    public bool IsRunning
    {
        get => this.isRunning;
        set
        {
            if (this.SetProperty(ref this.isRunning, value))
            {
                this.AskCommand.CanExecute = !value;
                this.StopCommand.CanExecute = value;
                this.flush.Change(value ? FlushInterval : Timeout.InfiniteTimeSpan, value ? FlushInterval : Timeout.InfiniteTimeSpan);
                if (!value)
                {
                    this.Flush();
                }
            }
        }
    }

    [DataMember]
    public AsyncCommand AskCommand { get; }

    [DataMember]
    public AsyncCommand StopCommand { get; }

    [DataMember]
    public AsyncCommand ClearCommand { get; }

    [DataMember]
    public AsyncCommand DetachCommand { get; }

    [DataMember]
    public AsyncCommand OpenCommand { get; }

    /// <summary>Adds text to the transcript; it shows at the next flush.</summary>
    public void Append(string text)
    {
        lock (this.gate)
        {
            this.transcript.Append(text);
            if (this.transcript.Length > MaxTranscript)
            {
                this.transcript.Remove(0, this.transcript.Length - MaxTranscript);
            }

            this.dirty = true;
        }

        if (!this.isRunning)
        {
            this.Flush();
        }
    }

    /// <summary>Empties the transcript.</summary>
    public void ClearTranscript()
    {
        lock (this.gate)
        {
            this.transcript.Clear();
            this.dirty = true;
        }

        this.Flush();
    }

    /// <summary>Sends the transcript to the window if it changed.</summary>
    public void Flush()
    {
        string text;
        lock (this.gate)
        {
            if (!this.dirty)
            {
                return;
            }

            this.dirty = false;
            text = this.transcript.ToString();
        }

        this.Transcript = text;
    }

    public void Dispose()
    {
        this.flush.Dispose();
    }
}

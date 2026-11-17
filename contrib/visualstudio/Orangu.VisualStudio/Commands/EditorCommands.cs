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
using System.Threading;
using System.Threading.Tasks;
using Microsoft.VisualStudio.Extensibility;
using Microsoft.VisualStudio.Extensibility.Commands;
using Microsoft.VisualStudio.Extensibility.Editor;
using Orangu.Core;
using Orangu.VisualStudio.ToolWindow;

namespace Orangu.VisualStudio.Commands;

/// <summary>The orangu group in the code editor's right-click menu.</summary>
internal static class EditorCommands
{
    // vsshlids.h: guidSHLMainMenu and IDM_VS_CTXT_CODEWIN, the code window's
    // context menu.
    private static readonly Guid ShellMainMenu = new("d309f791-903f-11d0-9efc-00a0c911004f");
    private const uint CodeWindowContextMenu = 0x040D;

    [VisualStudioContribution]
    public static CommandGroupConfiguration Group => new(GroupPlacement.VsctParent(ShellMainMenu, CodeWindowContextMenu, priority: 0x0600))
    {
        Children = [GroupChild.Command<AskAboutSelectionCommand>(), GroupChild.Command<ReviewFileCommand>()],
    };

    /// <summary>Only with a document in the editor.</summary>
    public static ActivationConstraint WithAnEditor { get; } =
        ActivationConstraint.ClientContext(ClientContextKey.Shell.ActiveEditorContentType, ".+");

    /// <summary>The selection, or the caret's line when nothing is selected.</summary>
    public static EditorSelection? SelectionOf(ITextViewSnapshot view)
    {
        if (string.IsNullOrEmpty(view.FilePath))
        {
            return null;
        }

        ITextDocumentSnapshot document = view.Document;
        Selection selection = view.Selection;
        int start = selection.Start.Offset;
        int end = selection.End.Offset;
        if (selection.IsEmpty)
        {
            ITextDocumentSnapshotLine caretLine = document.GetLineFromPosition(start);
            int line = caretLine.LineNumber + 1;
            return new EditorSelection(view.FilePath, line, line, caretLine.Text.CopyToString());
        }

        // A selection that ends at the start of a line does not take that line.
        int firstLine = document.GetLineNumberFromPosition(start) + 1;
        int lastLine = document.GetLineNumberFromPosition(Math.Max(start, end - 1)) + 1;
        return new EditorSelection(view.FilePath, firstLine, lastLine, selection.Extent.CopyToString());
    }
}

/// <summary>Attaches the selection to the tool window's next prompt.</summary>
[VisualStudioContribution]
internal sealed class AskAboutSelectionCommand : Command
{
    private readonly OranguSession session;

    public AskAboutSelectionCommand(OranguSession session)
    {
        this.session = session;
    }

    /// <inheritdoc />
    public override CommandConfiguration CommandConfiguration => new("%Orangu.AskCommand.DisplayName%")
    {
        Icon = new(ImageMoniker.KnownValues.Comment, IconSettings.IconAndText),
        EnabledWhen = EditorCommands.WithAnEditor,
    };

    /// <inheritdoc />
    public override async Task ExecuteCommandAsync(IClientContext context, CancellationToken cancellationToken)
    {
        EditorSelection? selection;
        using (ITextViewSnapshot? view = await context.GetActiveTextViewAsync(cancellationToken))
        {
            selection = view is null ? null : EditorCommands.SelectionOf(view);
        }

        string? workspace = await this.session.WorkspaceAsync(context, cancellationToken);
        if (selection is null || workspace is null)
        {
            return;
        }

        this.session.Attach(workspace, selection);
        await this.Extensibility.Shell().ShowToolWindowAsync<OranguToolWindow>(activate: true, cancellationToken);
    }
}

/// <summary>Asks orangu to review the file in the editor.</summary>
[VisualStudioContribution]
internal sealed class ReviewFileCommand : Command
{
    private readonly OranguSession session;

    public ReviewFileCommand(OranguSession session)
    {
        this.session = session;
    }

    /// <inheritdoc />
    public override CommandConfiguration CommandConfiguration => new("%Orangu.ReviewCommand.DisplayName%")
    {
        Icon = new(ImageMoniker.KnownValues.CodeReview, IconSettings.IconAndText),
        EnabledWhen = EditorCommands.WithAnEditor,
    };

    /// <inheritdoc />
    public override async Task ExecuteCommandAsync(IClientContext context, CancellationToken cancellationToken)
    {
        string? filePath;
        bool unsaved;
        using (ITextViewSnapshot? view = await context.GetActiveTextViewAsync(cancellationToken))
        {
            filePath = view?.FilePath;
            unsaved = view?.Document.IsDirty ?? false;
        }

        string? workspace = await this.session.WorkspaceAsync(context, cancellationToken);
        if (string.IsNullOrEmpty(filePath) || workspace is null)
        {
            return;
        }

        await this.Extensibility.Shell().ShowToolWindowAsync<OranguToolWindow>(activate: true, cancellationToken);
        if (unsaved)
        {
            // orangu reads the file from disk.
            this.session.Data.Append("(The file has unsaved changes; orangu reviews the saved version.)\n\n");
        }

        await this.session.ReviewAsync(workspace, filePath, cancellationToken);
    }
}

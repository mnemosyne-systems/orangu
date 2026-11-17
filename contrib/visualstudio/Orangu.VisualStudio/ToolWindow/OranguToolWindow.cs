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

using System.Threading;
using System.Threading.Tasks;
using Microsoft.VisualStudio.Extensibility;
using Microsoft.VisualStudio.Extensibility.ToolWindows;
using Microsoft.VisualStudio.RpcContracts.RemoteUI;

namespace Orangu.VisualStudio.ToolWindow;

/// <summary>The orangu tool window: a prompt box and the answers.</summary>
[VisualStudioContribution]
internal sealed class OranguToolWindow : Microsoft.VisualStudio.Extensibility.ToolWindows.ToolWindow
{
    private readonly OranguSession session;

    public OranguToolWindow(OranguSession session)
    {
        this.session = session;
        this.Title = "orangu";
    }

    /// <inheritdoc />
    public override ToolWindowConfiguration ToolWindowConfiguration => new()
    {
        // Beside Solution Explorer, where a chat panel is usually found.
        Placement = ToolWindowPlacement.DockedTo(SolutionExplorer),
        DockDirection = Dock.Bottom,
    };

    /// <inheritdoc />
    public override Task<IRemoteUserControl> GetContentAsync(CancellationToken cancellationToken)
    {
        return Task.FromResult<IRemoteUserControl>(new OranguToolWindowContent(this.session.Data));
    }

    private static readonly System.Guid SolutionExplorer = new("3ae79031-e1bc-11d0-8f78-00a0c9110057");
}

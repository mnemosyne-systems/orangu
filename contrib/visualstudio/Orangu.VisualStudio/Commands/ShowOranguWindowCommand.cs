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
using Microsoft.VisualStudio.Extensibility.Commands;
using Orangu.VisualStudio.ToolWindow;

namespace Orangu.VisualStudio.Commands;

/// <summary>View &gt; Other Windows &gt; orangu, and the same under Tools.</summary>
[VisualStudioContribution]
internal sealed class ShowOranguWindowCommand : Command
{
    /// <inheritdoc />
    public override CommandConfiguration CommandConfiguration => new("%Orangu.ShowWindowCommand.DisplayName%")
    {
        Icon = new(ImageMoniker.KnownValues.ToolWindow, IconSettings.IconAndText),
        Placements = [CommandPlacement.KnownPlacements.ViewOtherWindowsMenu, CommandPlacement.KnownPlacements.ToolsMenu],
    };

    /// <inheritdoc />
    public override Task ExecuteCommandAsync(IClientContext context, CancellationToken cancellationToken)
    {
        return this.Extensibility.Shell().ShowToolWindowAsync<OranguToolWindow>(activate: true, cancellationToken);
    }
}

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

namespace Orangu.VisualStudio.Commands;

/// <summary>
/// Tools &gt; Open orangu: orangu itself in its own window, in the solution
/// folder, with everything it does in a terminal.
/// </summary>
[VisualStudioContribution]
internal sealed class OpenOranguCommand : Command
{
    private readonly OranguSession session;

    public OpenOranguCommand(OranguSession session)
    {
        this.session = session;
    }

    /// <inheritdoc />
    public override CommandConfiguration CommandConfiguration => new("%Orangu.OpenCommand.DisplayName%")
    {
        TooltipText = "%Orangu.OpenCommand.Tooltip%",
        Icon = new(ImageMoniker.KnownValues.Console, IconSettings.IconAndText),
        Placements = [CommandPlacement.KnownPlacements.ToolsMenu],
    };

    /// <inheritdoc />
    public override Task ExecuteCommandAsync(IClientContext context, CancellationToken cancellationToken)
    {
        return this.session.OpenInteractiveAsync(context, cancellationToken);
    }
}

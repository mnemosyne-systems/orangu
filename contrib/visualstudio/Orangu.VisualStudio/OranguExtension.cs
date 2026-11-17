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

using Microsoft.Extensions.DependencyInjection;
using Microsoft.VisualStudio.Extensibility;

namespace Orangu.VisualStudio;

/// <summary>
/// orangu in Visual Studio. The extension does no inference and keeps no
/// agent of its own: it runs the orangu binary, which talks to
/// orangu-server or orangu-coordinator as orangu.conf says.
/// </summary>
[VisualStudioContribution]
internal sealed class OranguExtension : Extension
{
    /// <inheritdoc />
    public override ExtensionConfiguration ExtensionConfiguration => new()
    {
        Metadata = new(
            id: "Orangu.VisualStudio.8f0c2b7e-3a51-4d0e-9c47-6f2d1e9b5a30",
            version: this.ExtensionAssemblyVersion,
            publisherName: "The orangu community",
            displayName: "orangu",
            description: "Ask orangu about your code from Visual Studio, or open orangu in the solution folder.")
        {
            MoreInfo = "https://github.com/mnemosyne-systems/orangu",
            Tags = ["AI", "LLM", "orangu"],
        },
    };

    /// <inheritdoc />
    protected override void InitializeServices(IServiceCollection serviceCollection)
    {
        base.InitializeServices(serviceCollection);

        // One session for the tool window and every command that feeds it.
        serviceCollection.AddSingleton<OranguSession>();
    }
}

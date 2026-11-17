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

using Microsoft.VisualStudio.Extensibility;
using Microsoft.VisualStudio.Extensibility.Settings;

// The settings API is marked preview in the 17.14 SDK. It is what Tools >
// Options uses for out-of-process extensions, and if reading it ever fails
// the extension searches the PATH as it does for an empty setting.
#pragma warning disable VSEXTPREVIEW_SETTINGS

namespace Orangu.VisualStudio;

/// <summary>
/// Tools &gt; Options &gt; orangu. There is one setting: everything else
/// (the server, the model, the tools) comes from orangu.conf.
/// </summary>
internal static class OranguSettings
{
    /// <summary>The page in Tools &gt; Options.</summary>
    [VisualStudioContribution]
    internal static SettingCategory Category { get; } = new("orangu", "%Orangu.Settings.Category%");

    /// <summary>The orangu binary, or the folder it is in. Empty searches the PATH.</summary>
    [VisualStudioContribution]
    internal static Setting.String BinaryPath { get; } = new("binaryPath", "%Orangu.Settings.BinaryPath%", Category, defaultValue: string.Empty)
    {
        Description = "%Orangu.Settings.BinaryPath.Description%",
    };
}

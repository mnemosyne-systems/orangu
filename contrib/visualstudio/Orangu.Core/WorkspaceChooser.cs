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
using System.IO;

namespace Orangu.Core;

/// <summary>The folder orangu runs in, given what Visual Studio has open.</summary>
public static class WorkspaceChooser
{
    /// <summary>
    /// The solution's folder; else, for a single file opened on its own, the
    /// Git repository around it; else its folder. Null with nothing open.
    /// </summary>
    public static string? Choose(string? solutionDirectory, string? activeDocument, Func<string, bool> exists)
    {
        if (!string.IsNullOrWhiteSpace(solutionDirectory) && exists(solutionDirectory))
        {
            return solutionDirectory;
        }

        if (string.IsNullOrWhiteSpace(activeDocument))
        {
            return null;
        }

        string? folder = Path.GetDirectoryName(activeDocument);
        for (string? directory = folder; directory is not null; directory = Path.GetDirectoryName(directory))
        {
            // A worktree has a .git file rather than a folder; either counts.
            if (exists(Path.Combine(directory, ".git")))
            {
                return directory;
            }
        }

        return folder;
    }
}

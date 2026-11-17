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
using System.Collections.Generic;
using System.IO;

namespace Orangu.Core;

/// <summary>
/// Finds the orangu binary: the path set in the extension's settings, or
/// else <c>orangu</c> on the PATH, or else where <c>cargo install</c> puts it.
/// </summary>
public static class OranguLocator
{
    /// <summary>Locates orangu on this machine.</summary>
    /// <param name="configured">The settings path; empty means search.</param>
    public static LocateResult Locate(string? configured)
    {
        return Locate(
            configured,
            Environment.GetEnvironmentVariable("PATH"),
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            OperatingSystem.IsWindows(),
            File.Exists);
    }

    /// <summary>
    /// The same search, with everything it reads from the machine passed in.
    /// </summary>
    public static LocateResult Locate(
        string? configured,
        string? pathVariable,
        string? home,
        bool windows,
        Func<string, bool> exists)
    {
        string name = windows ? "orangu.exe" : "orangu";
        configured = configured?.Trim().Trim('"');
        if (!string.IsNullOrEmpty(configured))
        {
            // A folder is taken to mean the binary inside it.
            foreach (string candidate in new[] { configured, Path.Combine(configured, name) })
            {
                if (exists(candidate))
                {
                    return LocateResult.Found(candidate);
                }
            }

            return LocateResult.Missing(
                $"The orangu path in Tools > Options > orangu does not exist: {configured}");
        }

        foreach (string directory in SearchDirectories(pathVariable, home, windows))
        {
            string candidate = Path.Combine(directory, name);
            if (exists(candidate))
            {
                return LocateResult.Found(candidate);
            }
        }

        return LocateResult.Missing(
            $"Could not find {name} on the PATH. Install orangu, or set its path in Tools > Options > orangu.");
    }

    private static IEnumerable<string> SearchDirectories(string? pathVariable, string? home, bool windows)
    {
        char separator = windows ? ';' : ':';
        foreach (string entry in (pathVariable ?? string.Empty).Split(separator))
        {
            string directory = entry.Trim().Trim('"');
            if (directory.Length > 0)
            {
                yield return directory;
            }
        }

        // Visual Studio may have been started before ~/.cargo/bin was added
        // to the PATH, so `cargo install --path .` is looked for directly.
        if (!string.IsNullOrEmpty(home))
        {
            yield return Path.Combine(home, ".cargo", "bin");
        }
    }
}

/// <summary>Where orangu is, or why it could not be found.</summary>
public sealed class LocateResult
{
    private LocateResult(string? path, string? error)
    {
        this.Path = path;
        this.Error = error;
    }

    /// <summary>The binary's path, when it was found.</summary>
    public string? Path { get; }

    /// <summary>What to tell the user, when it was not.</summary>
    public string? Error { get; }

    internal static LocateResult Found(string path) => new(path, null);

    internal static LocateResult Missing(string error) => new(null, error);
}

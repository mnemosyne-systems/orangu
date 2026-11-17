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
using System.Linq;
using System.Reflection;
using System.Runtime.Serialization;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Xml.Linq;
using Microsoft.VisualStudio.Extensibility.UI;
using Orangu.VisualStudio.ToolWindow;
using Xunit;

namespace Orangu.VisualStudio.Tests;

/// <summary>
/// What Visual Studio only reads once the extension is running, where a
/// mistake shows up as an empty window or a raw %key% in a menu.
/// </summary>
public class RunTimeResourceTests
{
    private static readonly Assembly Extension = typeof(OranguToolWindowContent).Assembly;

    private static string ExtensionProject
    {
        get
        {
            for (var directory = new DirectoryInfo(AppContext.BaseDirectory); directory is not null; directory = directory.Parent)
            {
                string project = Path.Combine(directory.FullName, "Orangu.VisualStudio");
                if (File.Exists(Path.Combine(project, "Orangu.VisualStudio.csproj")))
                {
                    return project;
                }
            }

            throw new InvalidOperationException("Orangu.VisualStudio is not above the test output");
        }
    }

    [Fact]
    public void The_window_xaml_is_embedded_under_the_name_remote_ui_looks_for()
    {
        Assert.Contains(typeof(OranguToolWindowContent).FullName + ".xaml", Extension.GetManifestResourceNames());
    }

    [Fact]
    public void Every_binding_in_the_window_is_a_data_member()
    {
        XDocument xaml = LoadWindowXaml();
        Assert.Equal("DataTemplate", xaml.Root!.Name.LocalName);

        var members = typeof(OranguToolWindowData)
            .GetProperties(BindingFlags.Public | BindingFlags.Instance)
            .Where(property => property.GetCustomAttribute<DataMemberAttribute>() is not null)
            .Select(property => property.Name)
            .ToHashSet();
        var bound = Bindings(xaml).ToList();

        Assert.NotEmpty(bound);
        Assert.All(bound, path => Assert.Contains(path, members));
    }

    [Fact]
    public void Every_button_command_is_bound_once()
    {
        var commands = typeof(OranguToolWindowData)
            .GetProperties(BindingFlags.Public | BindingFlags.Instance)
            .Where(property => property.PropertyType == typeof(AsyncCommand))
            .Select(property => property.Name)
            .OrderBy(name => name);
        var bound = LoadWindowXaml().Descendants()
            .Select(element => (string?)element.Attribute("Command"))
            .Where(value => value is not null)
            .Select(value => BindingPath.Match(value!).Groups[1].Value)
            .OrderBy(name => name);

        Assert.Equal(commands, bound);
    }

    [Fact]
    public void Every_string_the_manifest_names_has_a_text()
    {
        string folder = Path.Combine(ExtensionProject, ".vsextension");
        using JsonDocument strings = JsonDocument.Parse(File.ReadAllText(Path.Combine(folder, "string-resources.json")));
        var known = strings.RootElement.EnumerateObject().Select(property => property.Name).ToHashSet();

        // What the build generated from the [VisualStudioContribution]s, in
        // the extension's own output (the newest, whichever configuration).
        string manifest = Directory
            .EnumerateFiles(Path.Combine(ExtensionProject, "bin"), "extension.json", SearchOption.AllDirectories)
            .OrderByDescending(File.GetLastWriteTimeUtc)
            .First();
        string generated = string.Concat(
            Directory.EnumerateFiles(Path.GetDirectoryName(manifest)!, "*.json")
                .Where(file => !file.EndsWith("string-resources.json", StringComparison.Ordinal))
                .Select(File.ReadAllText));
        var used = Regex.Matches(generated, "%(Orangu\\.[A-Za-z.]+)%").Select(match => match.Groups[1].Value).ToHashSet();

        Assert.NotEmpty(used);
        Assert.Empty(used.Except(known));
        Assert.Empty(known.Except(used));
    }

    private static readonly Regex BindingPath = new(@"^\{Binding\s+(?:Path=)?([A-Za-z]+)");

    private static IEnumerable<string> Bindings(XDocument xaml)
    {
        return xaml.Descendants()
            .SelectMany(element => element.Attributes())
            .Select(attribute => BindingPath.Match(attribute.Value))
            .Where(match => match.Success)
            .Select(match => match.Groups[1].Value);
    }

    private static XDocument LoadWindowXaml()
    {
        using Stream stream = Extension.GetManifestResourceStream(typeof(OranguToolWindowContent).FullName + ".xaml")!;
        return XDocument.Load(stream);
    }
}

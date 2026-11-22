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

package ai.mnemosyne_systems.orangu

import ai.mnemosyne_systems.orangu.core.interactive
import com.intellij.openapi.components.Service
import com.intellij.openapi.project.Project
import com.intellij.openapi.util.io.FileUtil
import com.intellij.openapi.wm.ToolWindowManager
import com.intellij.ui.content.Content
import org.jetbrains.plugins.terminal.TerminalTabState
import org.jetbrains.plugins.terminal.TerminalToolWindowFactory
import org.jetbrains.plugins.terminal.TerminalToolWindowManager

/**
 * orangu's own interface in the IDE's terminal: `orangu -w <workspace>` as
 * the tab's program, so it ends with orangu rather than leaving a shell.
 */
object OranguTerminal {
    /** Opens orangu for the workspace of `file`, or shows the one that is already running there. */
    fun open(project: Project, file: String?) {
        val target = OranguService.getInstance(project).resolve(file, inWindow = false) ?: return
        val toolWindow = ToolWindowManager.getInstance(project)
            .getToolWindow(TerminalToolWindowFactory.TOOL_WINDOW_ID) ?: return
        val tabs = project.getService(Tabs::class.java)

        val running = tabs.byWorkspace[target.workspace]?.takeIf { content ->
            content.manager != null && TerminalToolWindowManager.findWidgetByContent(content)?.ttyConnector?.isConnected == true
        }
        if (running != null) {
            toolWindow.activate {
                toolWindow.contentManager.setSelectedContent(running, true)
            }
            return
        }

        val workspace = FileUtil.toSystemDependentName(target.workspace)
        val state = TerminalTabState().apply {
            myTabName = "orangu"
            myWorkingDirectory = workspace
            myShellCommand = interactive(target.binary, workspace).command
        }
        val manager = TerminalToolWindowManager.getInstance(project)
        manager.createNewSession(manager.terminalRunner, state)
        // The new tab is the selected one.
        toolWindow.contentManager.selectedContent?.let { tabs.byWorkspace[target.workspace] = it }
        toolWindow.activate(null)
    }

    /** The orangu tab of each workspace in a project. */
    @Service(Service.Level.PROJECT)
    class Tabs {
        val byWorkspace = mutableMapOf<String, Content>()
    }
}

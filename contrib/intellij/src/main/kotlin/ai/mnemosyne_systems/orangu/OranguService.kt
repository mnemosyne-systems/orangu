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

import ai.mnemosyne_systems.orangu.core.CommandLine
import ai.mnemosyne_systems.orangu.core.DEFAULT_SELECTION_QUESTION
import ai.mnemosyne_systems.orangu.core.EditorSelection
import ai.mnemosyne_systems.orangu.core.Located
import ai.mnemosyne_systems.orangu.core.OneShotRun
import ai.mnemosyne_systems.orangu.core.Outcome
import ai.mnemosyne_systems.orangu.core.REVIEW_CHANGES
import ai.mnemosyne_systems.orangu.core.Review
import ai.mnemosyne_systems.orangu.core.RunResult
import ai.mnemosyne_systems.orangu.core.Transcript
import ai.mnemosyne_systems.orangu.core.aboutSelection
import ai.mnemosyne_systems.orangu.core.chooseWorkspace
import ai.mnemosyne_systems.orangu.core.oneShot
import ai.mnemosyne_systems.orangu.core.reviewFile
import ai.mnemosyne_systems.orangu.core.selectionLabel
import com.intellij.execution.configurations.GeneralCommandLine
import com.intellij.notification.NotificationAction
import com.intellij.notification.NotificationGroupManager
import com.intellij.notification.NotificationType
import com.intellij.openapi.Disposable
import com.intellij.openapi.application.ApplicationManager
import com.intellij.openapi.components.Service
import com.intellij.openapi.fileEditor.FileEditorManager
import com.intellij.openapi.options.ShowSettingsUtil
import com.intellij.openapi.project.Project
import com.intellij.openapi.roots.ProjectRootManager
import com.intellij.openapi.util.io.FileUtil
import com.intellij.openapi.vfs.VirtualFile
import com.intellij.openapi.wm.ToolWindowManager
import java.io.File

/**
 * The orangu tool window's state and what it does, one per project: the
 * prompts and answers, the attached selection, and the run that is going.
 * Everything here is called on the UI thread.
 */
@Service(Service.Level.PROJECT)
class OranguService(private val project: Project) : Disposable {
    val transcript = Transcript()

    private var running: OneShotRun? = null
    private var attached: EditorSelection? = null
    private val fillListeners = mutableListOf<(String) -> Unit>()

    /** A question for the prompt box that no tool window has taken yet. */
    private var pendingFill: String? = null

    /** Starts orangu; tests put their own in. */
    internal var startProcess: (CommandLine) -> Process = ::startWithIde

    val isRunning: Boolean get() = running != null

    /** Asks orangu; with a selection attached, about that selection. */
    fun ask(text: String) {
        val selection = attached
        if (text.isBlank() && selection == null) {
            return
        }
        val target = resolve(selection?.filePath) ?: return
        var prompt = text.trim()
        var shown = prompt
        if (selection != null) {
            prompt = aboutSelection(target.workspace, selection, text)
            val question = text.trim().ifEmpty { DEFAULT_SELECTION_QUESTION }
            shown = "$question\n(${selectionLabel(target.workspace, selection)})"
            removeSelection()
        }
        run(target, prompt, shown)
    }

    /** orangu's review of everything the branch changes. */
    fun reviewChanges() {
        val target = resolve(null) ?: return
        run(target, REVIEW_CHANGES, REVIEW_CHANGES)
    }

    /** orangu's review of one file, as it is on disk. */
    fun reviewFile(file: VirtualFile, unsaved: Boolean) {
        val target = resolve(file.path) ?: return
        when (val review = reviewFile(target.workspace, file.path)) {
            is Review.Refused -> {
                transcript.fail(review.error)
                notify(review.error, NotificationType.WARNING)
            }
            is Review.Prompt -> {
                // orangu reads the file from disk and Git, not from the editor.
                val shown = if (unsaved) {
                    "${review.prompt}\n(the file has unsaved changes; orangu reviews it as it was last saved)"
                } else {
                    review.prompt
                }
                run(target, review.prompt, shown)
            }
        }
    }

    /** Attaches a selection to the next prompt, and puts a question in the box if it is empty. */
    fun attachSelection(selection: EditorSelection) {
        val workspace = workspaceFor(selection.filePath) ?: ""
        attached = selection
        transcript.attach(selectionLabel(workspace, selection))
        // The tool window may not have been opened yet; it gets the question once it is.
        pendingFill = DEFAULT_SELECTION_QUESTION
        deliverFill()
    }

    fun removeSelection() {
        attached = null
        transcript.attach(null)
    }

    fun stop() {
        running?.stop()
    }

    fun clear() {
        transcript.clear()
    }

    /** Calls `fill` with questions for the prompt box, until the returned function is called. */
    fun onFill(fill: (String) -> Unit): () -> Unit {
        fillListeners += fill
        deliverFill()
        return { fillListeners -= fill }
    }

    /** Shows the orangu tool window. */
    fun showToolWindow() {
        ToolWindowManager.getInstance(project).getToolWindow(TOOL_WINDOW_ID)?.activate(null, false)
    }

    /** The folder orangu runs in for `file`, or for the file in the editor. */
    fun workspaceFor(file: String?): String? = chooseWorkspace(
        project.basePath,
        ProjectRootManager.getInstance(project).contentRoots.map { it.path },
        file ?: FileEditorManager.getInstance(project).selectedFiles.firstOrNull()?.path,
    ) { File(it).exists() }

    override fun dispose() {
        running?.stop()
    }

    private fun run(target: Target, prompt: String, shown: String) {
        if (running != null) {
            notify("orangu is still answering; stop it first.", NotificationType.INFORMATION)
            return
        }
        val line = oneShot(target.binary, FileUtil.toSystemDependentName(target.workspace), prompt)
        val application = ApplicationManager.getApplication()
        val onUi = { action: () -> Unit -> application.invokeLater({ action() }, project.disposed) }
        val run = OneShotRun(
            line,
            startProcess,
            answer = { text -> onUi { transcript.answer(text) } },
            note = { text -> onUi { transcript.note(text) } },
        )
        running = run
        val started = System.currentTimeMillis()
        transcript.begin(shown)
        application.executeOnPooledThread {
            // Whatever happens, the run ends, so the window is not left busy.
            val result = try {
                run.run()
            } catch (e: Throwable) {
                RunResult(Outcome.NOT_STARTED, error = e.message ?: e.toString())
            }
            onUi {
                running = null
                transcript.end(result, System.currentTimeMillis() - started)
            }
        }
    }

    /**
     * The binary and workspace, or null after showing why there are none: in
     * a notification, and on the tool window's status line when `inWindow`.
     */
    internal fun resolve(file: String?, inWindow: Boolean = true): Target? {
        val located = OranguSettings.getInstance().locateBinary()
        if (located is Located.Missing) {
            if (inWindow) {
                transcript.fail(located.error)
            }
            NotificationGroupManager.getInstance().getNotificationGroup(NOTIFICATIONS)
                .createNotification(located.error, NotificationType.ERROR)
                .addAction(
                    NotificationAction.createSimpleExpiring("Open settings") {
                        ShowSettingsUtil.getInstance().showSettingsDialog(project, OranguConfigurable::class.java)
                    },
                )
                .notify(project)
            return null
        }
        val workspace = workspaceFor(file)
        if (workspace == null) {
            val error = "Open a project or a file for orangu to work in."
            if (inWindow) {
                transcript.fail(error)
            }
            notify(error, NotificationType.WARNING)
            return null
        }
        return Target((located as Located.Found).path, workspace)
    }

    private fun deliverFill() {
        val text = pendingFill ?: return
        if (fillListeners.isNotEmpty()) {
            pendingFill = null
            fillListeners.toList().forEach { it(text) }
        }
    }

    private fun notify(message: String, type: NotificationType) {
        NotificationGroupManager.getInstance().getNotificationGroup(NOTIFICATIONS)
            .createNotification(message, type)
            .notify(project)
    }

    internal data class Target(val binary: String, val workspace: String)

    companion object {
        const val TOOL_WINDOW_ID = "orangu"
        const val NOTIFICATIONS = "orangu"

        fun getInstance(project: Project): OranguService = project.getService(OranguService::class.java)
    }
}

/**
 * Starts a program through the IDE, which quotes the arguments the way
 * Windows programs read them back and gives it the environment of the
 * user's shell (on macOS, an IDE started from the Dock has a bare PATH
 * otherwise).
 */
internal fun startWithIde(line: CommandLine): Process =
    GeneralCommandLine(line.command)
        .withWorkDirectory(line.workingDirectory)
        .withEnvironment(line.environment)
        .withParentEnvironmentType(GeneralCommandLine.ParentEnvironmentType.CONSOLE)
        .withCharset(Charsets.UTF_8)
        .createProcess()

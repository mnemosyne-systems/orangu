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

import ai.mnemosyne_systems.orangu.core.DEFAULT_SELECTION_QUESTION
import ai.mnemosyne_systems.orangu.core.EditorSelection
import ai.mnemosyne_systems.orangu.core.Entry
import ai.mnemosyne_systems.orangu.core.aboutSelection
import com.intellij.openapi.actionSystem.ActionManager
import com.intellij.openapi.actionSystem.DefaultActionGroup
import com.intellij.openapi.application.WriteAction
import com.intellij.openapi.command.WriteCommandAction
import com.intellij.openapi.fileEditor.FileDocumentManager
import com.intellij.openapi.options.Configurable
import com.intellij.openapi.util.Disposer
import com.intellij.openapi.util.io.FileUtil
import com.intellij.openapi.vfs.LocalFileSystem
import com.intellij.openapi.vfs.VfsUtil
import com.intellij.openapi.vfs.VirtualFile
import com.intellij.openapi.wm.ToolWindowEP
import com.intellij.testFramework.PlatformTestUtil
import com.intellij.testFramework.fixtures.BasePlatformTestCase
import java.io.File

/**
 * The plugin in a running IDE: what it registers, and each action and the
 * tool window end to end, against the stand-in for orangu.
 */
class PluginTest : BasePlatformTestCase() {
    private lateinit var service: OranguService
    private lateinit var binary: File
    private lateinit var workspace: String

    override fun setUp() {
        super.setUp()
        workspace = project.basePath!!
        File(workspace).mkdirs()
        // The setting has to name a file; the stand-in is run in its place.
        binary = FileUtil.createTempFile("orangu", "", true)
        OranguSettings.getInstance().path = binary.path
        service = OranguService.getInstance(project)
        service.startProcess = { line -> startWithIde(StandIn.instead(line)) }
    }

    override fun tearDown() {
        try {
            service.stop()
            waitForRun()
            service.removeSelection()
            service.clear()
            OranguSettings.getInstance().path = ""
        } catch (e: Throwable) {
            addSuppressedException(e)
        } finally {
            super.tearDown()
        }
    }

    fun `test the actions are registered and in their menus`() {
        val actions = ActionManager.getInstance()
        for (id in listOf("orangu.Open", "orangu.ReviewChanges", "orangu.AskAboutSelection", "orangu.ReviewFile")) {
            assertNotNull(id, actions.getAction(id))
        }
        assertTrue(children("ToolsMenu").contains(actions.getAction("orangu.ToolsMenu")))
        assertTrue(children("EditorPopupMenu").contains(actions.getAction("orangu.EditorPopup")))
    }

    fun `test the tool window and the settings page are registered`() {
        assertTrue(ToolWindowEP.EP_NAME.extensionList.any { it.id == "orangu" && it.anchor == "right" })
        assertTrue(Configurable.APPLICATION_CONFIGURABLE.extensionList.any { it.instanceClass == OranguConfigurable::class.java.name })
    }

    fun `test the setting is kept without the spaces around it`() {
        val settings = OranguSettings.getInstance()
        settings.path = "  /opt/orangu  "
        assertEquals("/opt/orangu", settings.path)
        val copy = OranguSettings()
        copy.loadState(settings.state)
        assertEquals("/opt/orangu", copy.path)
    }

    fun `test a prompt runs orangu in the project and the answer comes in`() {
        service.ask("  what is \"this\"?  ")
        waitForRun()
        val snapshot = service.transcript.snapshot()
        assertEquals(Entry(Entry.Kind.PROMPT, "what is \"this\"?"), snapshot.entries.first())
        val report = StandIn.parse(snapshot.entries[1].text)
        assertEquals(listOf("-w", FileUtil.toSystemDependentName(workspace), "-p", "what is \"this\"?"), report.args)
        assertEquals("1", report.noColor)
        assertTrue(report.stdinClosed)
        assertTrue(snapshot.status, snapshot.status.startsWith("Done in "))
    }

    fun `test a failure shows orangu's own message`() {
        service.ask("fail")
        waitForRun()
        assertEquals("orangu failed: error: the stand-in server is not running", service.transcript.snapshot().status)
    }

    fun `test Stop ends a run`() {
        service.ask("sleep")
        PlatformTestUtil.waitWithEventsDispatching(
            "the stand-in never started its child",
            { service.transcript.snapshot().status.contains("child") },
            30,
        )
        service.stop()
        waitForRun()
        assertEquals("Stopped", service.transcript.snapshot().status)
    }

    fun `test a second prompt while one runs is not started`() {
        service.ask("sleep")
        service.ask("another")
        assertEquals(1, service.transcript.snapshot().entries.count { it.kind == Entry.Kind.PROMPT })
    }

    fun `test a binary that is not there is reported`() {
        OranguSettings.getInstance().path = File(workspace, "nowhere/orangu").path
        service.ask("hi")
        assertFalse(service.isRunning)
        val status = service.transcript.snapshot().status
        assertTrue(status, status.startsWith("Settings > Tools > orangu names a file that does not exist"))
    }

    fun `test a selection is attached and asked about`() {
        val file = projectFile("src/sample.kt", "fun a() {\n    val x = 1\n    val y = 2\n}\n")
        myFixture.openFileInEditor(file)
        val document = myFixture.editor.document
        // Dragged over whole lines: the selection ends at the start of line 4.
        myFixture.editor.selectionModel.setSelection(document.getLineStartOffset(1), document.getLineStartOffset(3))
        myFixture.testAction(ActionManager.getInstance().getAction("orangu.AskAboutSelection"))
        assertEquals("src/sample.kt:2-3", service.transcript.snapshot().selection)

        service.ask("")
        waitForRun()
        val snapshot = service.transcript.snapshot()
        assertNull(snapshot.selection)
        assertEquals("$DEFAULT_SELECTION_QUESTION\n(src/sample.kt:2-3)", snapshot.entries.first().text)
        val selection = EditorSelection(file.path, 2, 3, "    val x = 1\n    val y = 2\n")
        assertEquals(aboutSelection(workspace, selection, ""), StandIn.parse(snapshot.entries[1].text).args[3])
    }

    fun `test with nothing selected the caret's line is attached`() {
        val file = projectFile("src/sample.kt", "fun a() {\n    val x = 1\n}\n")
        myFixture.openFileInEditor(file)
        myFixture.editor.caretModel.moveToOffset(myFixture.editor.document.getLineStartOffset(1) + 4)
        myFixture.testAction(ActionManager.getInstance().getAction("orangu.AskAboutSelection"))
        assertEquals("src/sample.kt:2", service.transcript.snapshot().selection)
    }

    fun `test a file review is auto_review of the file, escaped`() {
        myFixture.openFileInEditor(projectFile("app/[id]/page.kt", "val page = 1\n"))
        myFixture.testAction(ActionManager.getInstance().getAction("orangu.ReviewFile"))
        waitForRun()
        val snapshot = service.transcript.snapshot()
        assertEquals("/auto_review app/[[]id[]]/page.kt", snapshot.entries.first().text)
        assertEquals("/auto_review app/[[]id[]]/page.kt", StandIn.parse(snapshot.entries[1].text).args[3])
    }

    fun `test a file with unsaved changes is reviewed as saved, which is pointed out`() {
        val file = projectFile("src/main.kt", "val a = 1\n")
        myFixture.openFileInEditor(file)
        WriteCommandAction.runWriteCommandAction(project) { myFixture.editor.document.insertString(0, "// edited\n") }
        assertTrue(FileDocumentManager.getInstance().isFileModified(file))
        myFixture.testAction(ActionManager.getInstance().getAction("orangu.ReviewFile"))
        waitForRun()
        assertEquals(
            "/auto_review src/main.kt\n(the file has unsaved changes; orangu reviews it as it was last saved)",
            service.transcript.snapshot().entries.first().text,
        )
    }

    fun `test a file outside the project is not reviewed`() {
        val outside = File(FileUtil.createTempDirectory("elsewhere", null, true), "scratch.kt").apply { writeText("x\n") }
        service.reviewFile(LocalFileSystem.getInstance().refreshAndFindFileByIoFile(outside)!!, false)
        assertFalse(service.isRunning)
        assertTrue(service.transcript.snapshot().status.contains("outside the workspace"))
    }

    fun `test Review Changes is auto_review of the branch`() {
        myFixture.testAction(ActionManager.getInstance().getAction("orangu.ReviewChanges"))
        waitForRun()
        val snapshot = service.transcript.snapshot()
        assertEquals("/auto_review", snapshot.entries.first().text)
        assertEquals(listOf("-w", FileUtil.toSystemDependentName(workspace), "-p", "/auto_review"), StandIn.parse(snapshot.entries[1].text).args)
    }

    fun `test the tool window shows the run and its buttons follow it`() {
        val panel = OranguPanel(project).also { Disposer.register(testRootDisposable, it) }
        assertTrue(panel.ask.isEnabled)
        assertFalse(panel.stop.isEnabled)

        panel.prompt.text = "sleep"
        panel.ask.doClick()
        assertEquals("", panel.prompt.text)
        assertFalse(panel.ask.isEnabled)
        assertTrue(panel.stop.isEnabled)
        panel.stop.doClick()
        waitForRun()
        assertEquals("Stopped", panel.status.text)

        panel.prompt.text = "hello"
        panel.ask.doClick()
        waitForRun()
        val text = panel.transcriptView.text
        assertTrue(text, text.contains("sleep") && text.contains("hello") && text.contains("noColor 1"))
        assertTrue(panel.status.text, panel.status.text.startsWith("Done in "))
        assertTrue(panel.ask.isEnabled)

        panel.clear.doClick()
        assertEquals("", panel.transcriptView.text)
        assertEquals("", panel.status.text)
    }

    fun `test attaching a selection fills the prompt box, and Remove takes it off`() {
        val panel = OranguPanel(project).also { Disposer.register(testRootDisposable, it) }
        service.attachSelection(EditorSelection("$workspace/src/a.kt", 3, 9, "code"))
        assertEquals(DEFAULT_SELECTION_QUESTION, panel.prompt.text)
        assertEquals("src/a.kt:3-9", panel.attachment.text)
        panel.removeSelection.doClick()
        assertNull(service.transcript.snapshot().selection)
        assertEquals("", panel.attachment.text)
    }

    fun `test a question for the prompt box waits for the tool window to open`() {
        service.attachSelection(EditorSelection("$workspace/src/a.kt", 3, 3, "code"))
        val panel = OranguPanel(project).also { Disposer.register(testRootDisposable, it) }
        assertEquals(DEFAULT_SELECTION_QUESTION, panel.prompt.text)
        assertEquals("src/a.kt:3", panel.attachment.text)
    }

    private fun waitForRun() =
        PlatformTestUtil.waitWithEventsDispatching("orangu did not finish", { !service.isRunning }, 60)

    private fun children(group: String) =
        (ActionManager.getInstance().getAction(group) as DefaultActionGroup).childActionsOrStubs.toList()

    // Written through the IDE, so a file an earlier test wrote with other text
    // is not still in its cache (the light test project is shared).
    private fun projectFile(relative: String, text: String): VirtualFile = WriteAction.computeAndWait<VirtualFile, Throwable> {
        val folder = VfsUtil.createDirectories(File(workspace, relative).parent)
        val file = folder.findOrCreateChildData(this, relative.substringAfterLast('/'))
        VfsUtil.saveText(file, text)
        file
    }
}

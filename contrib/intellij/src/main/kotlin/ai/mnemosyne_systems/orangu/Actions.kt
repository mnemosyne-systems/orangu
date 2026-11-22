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

import ai.mnemosyne_systems.orangu.core.EditorSelection
import com.intellij.openapi.actionSystem.ActionUpdateThread
import com.intellij.openapi.actionSystem.AnActionEvent
import com.intellij.openapi.actionSystem.CommonDataKeys
import com.intellij.openapi.editor.Editor
import com.intellij.openapi.fileEditor.FileDocumentManager
import com.intellij.openapi.project.DumbAwareAction
import com.intellij.openapi.util.TextRange
import com.intellij.openapi.vfs.VirtualFile

/** Starts orangu's own interface in the IDE's terminal, in the workspace. */
class OpenOranguAction : DumbAwareAction() {
    override fun getActionUpdateThread() = ActionUpdateThread.BGT

    override fun update(e: AnActionEvent) {
        e.presentation.isEnabledAndVisible = e.project != null
    }

    override fun actionPerformed(e: AnActionEvent) {
        val project = e.project ?: return
        OranguTerminal.open(project, e.getData(CommonDataKeys.VIRTUAL_FILE)?.takeIf { it.isInLocalFileSystem }?.path)
    }
}

/** orangu's review of everything the branch changes, `/auto_review`. */
class ReviewChangesAction : DumbAwareAction() {
    override fun getActionUpdateThread() = ActionUpdateThread.BGT

    override fun update(e: AnActionEvent) {
        e.presentation.isEnabledAndVisible = e.project != null
    }

    override fun actionPerformed(e: AnActionEvent) {
        val service = OranguService.getInstance(e.project ?: return)
        service.showToolWindow()
        service.reviewChanges()
    }
}

/** Attaches the selection, or the caret's line, to the next prompt in the tool window. */
class AskAboutSelectionAction : DumbAwareAction() {
    override fun getActionUpdateThread() = ActionUpdateThread.BGT

    override fun update(e: AnActionEvent) {
        e.presentation.isEnabledAndVisible = e.project != null && e.getData(CommonDataKeys.EDITOR) != null &&
            localFile(e) != null
    }

    override fun actionPerformed(e: AnActionEvent) {
        val service = OranguService.getInstance(e.project ?: return)
        val editor = e.getData(CommonDataKeys.EDITOR) ?: return
        val file = localFile(e) ?: return
        service.attachSelection(selectionOf(editor, file))
        service.showToolWindow()
    }
}

/** orangu's review of the file in the editor, `/auto_review <file>`. */
class ReviewFileAction : DumbAwareAction() {
    override fun getActionUpdateThread() = ActionUpdateThread.BGT

    override fun update(e: AnActionEvent) {
        e.presentation.isEnabledAndVisible = e.project != null && localFile(e) != null
    }

    override fun actionPerformed(e: AnActionEvent) {
        val service = OranguService.getInstance(e.project ?: return)
        val file = localFile(e) ?: return
        service.showToolWindow()
        service.reviewFile(file, FileDocumentManager.getInstance().isFileModified(file))
    }
}

/** The file the action is about, when orangu can read it from disk. */
private fun localFile(e: AnActionEvent): VirtualFile? =
    e.getData(CommonDataKeys.VIRTUAL_FILE)?.takeIf { it.isInLocalFileSystem && !it.isDirectory }

/**
 * The selection, or with nothing selected the caret's line. A selection that
 * ends at the start of a line, as one made by dragging over whole lines
 * does, does not take in that last line.
 */
internal fun selectionOf(editor: Editor, file: VirtualFile): EditorSelection {
    val document = editor.document
    val model = editor.selectionModel
    if (!model.hasSelection()) {
        val line = document.getLineNumber(editor.caretModel.offset)
        val text = document.getText(TextRange(document.getLineStartOffset(line), document.getLineEndOffset(line)))
        return EditorSelection(file.path, line + 1, line + 1, text)
    }
    val start = document.getLineNumber(model.selectionStart)
    var end = document.getLineNumber(model.selectionEnd)
    if (end > start && model.selectionEnd == document.getLineStartOffset(end)) {
        end--
    }
    return EditorSelection(file.path, start + 1, end + 1, model.selectedText.orEmpty())
}

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

import ai.mnemosyne_systems.orangu.core.Change
import ai.mnemosyne_systems.orangu.core.Entry
import ai.mnemosyne_systems.orangu.core.Snapshot
import com.intellij.openapi.Disposable
import com.intellij.openapi.editor.colors.EditorColorsManager
import com.intellij.openapi.project.Project
import com.intellij.ui.JBColor
import com.intellij.ui.components.JBLabel
import com.intellij.ui.components.JBScrollPane
import com.intellij.ui.components.JBTextArea
import com.intellij.util.ui.JBUI
import com.intellij.util.ui.UIUtil
import com.intellij.util.ui.WrapLayout
import java.awt.BorderLayout
import java.awt.FlowLayout
import java.awt.event.KeyAdapter
import java.awt.event.KeyEvent
import javax.swing.BoxLayout
import javax.swing.JButton
import javax.swing.JPanel
import javax.swing.JTextPane
import javax.swing.text.SimpleAttributeSet
import javax.swing.text.StyleConstants

/**
 * The orangu tool window: the prompts and answers, the attached selection,
 * a prompt box, the buttons and a status line.
 */
class OranguPanel(private val project: Project) : JPanel(BorderLayout()), Disposable {
    private val service = OranguService.getInstance(project)

    internal val transcriptView = JTextPane().apply {
        isEditable = false
        border = JBUI.Borders.empty(6, 8)
    }
    internal val empty = JBLabel(
        "<html>Ask orangu about the project, or right-click in the editor to ask about a selection " +
            "or review a file. Each prompt runs <code>orangu -p</code> on its own; for a conversation, " +
            "press Open orangu.</html>",
    ).apply {
        foreground = UIUtil.getContextHelpForeground()
        border = JBUI.Borders.empty(8)
        verticalAlignment = JBLabel.TOP
    }
    internal val attachment = JBLabel()
    internal val removeSelection = JButton("Remove")
    private val attachmentRow = JPanel(BorderLayout(JBUI.scale(8), 0)).apply {
        border = JBUI.Borders.empty(2, 6)
        add(attachment, BorderLayout.CENTER)
        add(removeSelection, BorderLayout.EAST)
        isVisible = false
    }
    internal val prompt = JBTextArea(3, 20).apply {
        lineWrap = true
        wrapStyleWord = true
        emptyText.text = "Ask orangu (Enter to send, Shift+Enter for a new line)"
    }
    internal val ask = JButton("Ask")
    internal val stop = JButton("Stop")
    internal val clear = JButton("Clear")
    internal val reviewChanges = JButton("Review Changes")
    internal val open = JButton("Open orangu")
    internal val status = JBTextArea().apply {
        isEditable = false
        lineWrap = true
        wrapStyleWord = true
        isOpaque = false
        foreground = UIUtil.getContextHelpForeground()
        border = JBUI.Borders.empty(2, 2)
    }

    private val scroll = JBScrollPane(transcriptView)
    private val center = JPanel(BorderLayout())
    private val unlisten: () -> Unit
    private val unfill: () -> Unit

    init {
        // The buttons wrap onto a second line in a narrow window.
        val buttons = JPanel(WrapLayout(FlowLayout.LEFT, JBUI.scale(4), JBUI.scale(4))).apply {
            add(ask)
            add(stop)
            add(clear)
            add(reviewChanges)
            add(open)
        }
        val bottom = JPanel().apply {
            layout = BoxLayout(this, BoxLayout.Y_AXIS)
            border = JBUI.Borders.empty(4, 6, 6, 6)
            add(attachmentRow.also { it.alignmentX = LEFT_ALIGNMENT })
            add(JBScrollPane(prompt).also { it.alignmentX = LEFT_ALIGNMENT })
            add(buttons.also { it.alignmentX = LEFT_ALIGNMENT })
            add(status.also { it.alignmentX = LEFT_ALIGNMENT })
        }
        add(center, BorderLayout.CENTER)
        add(bottom, BorderLayout.SOUTH)

        ask.addActionListener { send() }
        stop.addActionListener { service.stop() }
        clear.addActionListener { service.clear() }
        reviewChanges.addActionListener { service.reviewChanges() }
        open.addActionListener { OranguTerminal.open(project, null) }
        removeSelection.addActionListener { service.removeSelection() }
        prompt.addKeyListener(object : KeyAdapter() {
            override fun keyPressed(e: KeyEvent) {
                if (e.keyCode == KeyEvent.VK_ENTER && !e.isShiftDown) {
                    e.consume()
                    send()
                } else if (e.keyCode == KeyEvent.VK_ENTER) {
                    e.consume()
                    prompt.replaceSelection("\n")
                }
            }
        })

        show(service.transcript.snapshot())
        unlisten = service.transcript.listen(::apply)
        unfill = service.onFill { question ->
            if (prompt.text.isBlank()) {
                prompt.text = question
                prompt.selectAll()
            }
            prompt.requestFocusInWindow()
        }
    }

    override fun dispose() {
        unlisten()
        unfill()
    }

    private fun send() {
        if (service.isRunning) {
            return
        }
        val text = prompt.text
        if (text.isBlank() && attachmentRow.isVisible.not()) {
            return
        }
        prompt.text = ""
        service.ask(text)
    }

    private fun apply(change: Change) {
        when (change) {
            is Change.Reset -> show(change.snapshot)
            is Change.Answer -> {
                append(change.text, answerStyle())
                scrollToEnd()
            }
        }
    }

    private fun show(snapshot: Snapshot) {
        val document = transcriptView.styledDocument
        document.remove(0, document.length)
        snapshot.entries.forEachIndexed { index, entry ->
            if (index > 0) {
                append("\n\n", answerStyle())
            }
            when (entry.kind) {
                Entry.Kind.PROMPT -> append(entry.text, promptStyle())
                Entry.Kind.ANSWER -> append(entry.text, answerStyle())
            }
        }
        center.removeAll()
        center.add(if (snapshot.entries.isEmpty()) empty else scroll, BorderLayout.CENTER)
        center.revalidate()
        center.repaint()
        scrollToEnd()

        attachmentRow.isVisible = snapshot.selection != null
        attachment.text = snapshot.selection.orEmpty()
        status.text = snapshot.status
        ask.isEnabled = !snapshot.busy
        stop.isEnabled = snapshot.busy
        reviewChanges.isEnabled = !snapshot.busy
    }

    private fun append(text: String, style: SimpleAttributeSet) {
        val document = transcriptView.styledDocument
        document.insertString(document.length, text, style)
    }

    private fun scrollToEnd() {
        transcriptView.caretPosition = transcriptView.document.length
    }

    private fun promptStyle() = SimpleAttributeSet().apply {
        StyleConstants.setBold(this, true)
        StyleConstants.setForeground(this, JBColor.namedColor("Link.activeForeground", JBUI.CurrentTheme.Link.Foreground.ENABLED))
    }

    private fun answerStyle() = SimpleAttributeSet().apply {
        val scheme = EditorColorsManager.getInstance().globalScheme
        StyleConstants.setFontFamily(this, scheme.editorFontName)
        StyleConstants.setFontSize(this, scheme.editorFontSize)
        StyleConstants.setForeground(this, UIUtil.getLabelForeground())
    }
}

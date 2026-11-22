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

import com.intellij.openapi.fileChooser.FileChooserDescriptor
import com.intellij.openapi.options.Configurable
import com.intellij.openapi.ui.TextFieldWithBrowseButton
import com.intellij.ui.components.JBLabel
import com.intellij.util.ui.FormBuilder
import com.intellij.util.ui.UIUtil
import javax.swing.JComponent
import javax.swing.JPanel

/** Settings > Tools > orangu. */
class OranguConfigurable : Configurable {
    private var field: TextFieldWithBrowseButton? = null

    override fun getDisplayName(): String = "orangu"

    override fun createComponent(): JComponent {
        val field = TextFieldWithBrowseButton()
        // The binary itself, or the folder it is in.
        field.addBrowseFolderListener(null, FileChooserDescriptor(true, true, false, false, false, false))
        this.field = field
        return FormBuilder.createFormBuilder()
            .addLabeledComponent("orangu binary:", field)
            .addComponentToRightColumn(
                JBLabel(
                    "<html>The orangu binary, or the folder it is in. Left empty, orangu is looked for on " +
                        "the PATH, then in ~/.cargo/bin. Everything else comes from orangu.conf.</html>",
                ).apply { componentStyle = UIUtil.ComponentStyle.SMALL },
            )
            .addComponentFillVertically(JPanel(), 0)
            .panel
    }

    override fun isModified(): Boolean = field?.text?.trim() != OranguSettings.getInstance().path

    override fun apply() {
        OranguSettings.getInstance().path = field?.text.orEmpty()
    }

    override fun reset() {
        field?.text = OranguSettings.getInstance().path
    }

    override fun disposeUIResources() {
        field = null
    }
}

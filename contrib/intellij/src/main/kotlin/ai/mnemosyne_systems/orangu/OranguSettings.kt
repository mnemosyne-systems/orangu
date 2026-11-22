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

import ai.mnemosyne_systems.orangu.core.Located
import ai.mnemosyne_systems.orangu.core.ThisMachine
import ai.mnemosyne_systems.orangu.core.locate
import com.intellij.openapi.application.ApplicationManager
import com.intellij.openapi.components.PersistentStateComponent
import com.intellij.openapi.components.RoamingType
import com.intellij.openapi.components.Service
import com.intellij.openapi.components.State
import com.intellij.openapi.components.Storage

/**
 * The plugin's one setting: where the orangu binary is. It belongs to this
 * machine, so it is not synced to others with the IDE's settings.
 */
@Service(Service.Level.APP)
@State(name = "OranguSettings", storages = [Storage(value = "orangu.xml", roamingType = RoamingType.DISABLED)])
class OranguSettings : PersistentStateComponent<OranguSettings.Data> {
    class Data {
        /** The binary, or the folder it is in; empty to look for it. */
        var path: String = ""
    }

    private var data = Data()

    var path: String
        get() = data.path
        set(value) {
            data.path = value.trim()
        }

    override fun getState(): Data = data

    override fun loadState(state: Data) {
        data = state
    }

    /** The binary to run, or why there is none. */
    fun locateBinary(): Located = locate(path, ThisMachine)

    companion object {
        fun getInstance(): OranguSettings = ApplicationManager.getApplication().getService(OranguSettings::class.java)
    }
}

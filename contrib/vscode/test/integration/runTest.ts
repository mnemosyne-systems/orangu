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

// Downloads VS Code (stable, or the version in VSCODE_VERSION), starts it
// with the extension and the test workspace, and runs ./suite in it.

import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { runTests } from '@vscode/test-electron';

async function main(): Promise<void> {
    const root = path.resolve(__dirname, '..', '..', '..');
    // A copy, so the tests can make it a Git repository and change it.
    const workspace = fs.mkdtempSync(path.join(os.tmpdir(), 'orangu-vscode-'));
    fs.cpSync(path.join(root, 'test', 'fixtures', 'workspace'), workspace, { recursive: true });
    try {
        await runTests({
            version: process.env.VSCODE_VERSION ?? 'stable',
            extensionDevelopmentPath: root,
            extensionTestsPath: path.join(__dirname, 'suite'),
            launchArgs: [workspace, '--disable-extensions', '--disable-workspace-trust', '--skip-welcome'],
            extensionTestsEnv: { ORANGU_TEST_ROOT: root },
        });
    } finally {
        fs.rmSync(workspace, { recursive: true, force: true });
    }
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});

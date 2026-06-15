@echo off
:: Copyright (C) 2026 The orangu community
::
:: This program is free software: you can redistribute it and/or modify
:: it under the terms of the GNU General Public License as published by
:: the Free Software Foundation, either version 3 of the License, or
:: (at your option) any later version.
::
:: This program is distributed in the hope that it will be useful,
:: but WITHOUT ANY WARRANTY; without even the implied warranty of
:: MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
:: GNU General Public License for more details.
::
:: You should have received a copy of the GNU General Public License
:: along with this program. If not, see <https://www.gnu.org/licenses/>.
:: Usage: curl -fsSL https://raw.githubusercontent.com/mnemosyne-systems/orangu/main/install.cmd -o install.cmd && install.cmd
:: Override install directory: set "INSTALL_DIR=C:\Tools" && install.cmd
setlocal EnableDelayedExpansion

set "REPO=mnemosyne-systems/orangu"
if not defined INSTALL_DIR set "INSTALL_DIR=%USERPROFILE%\.local\bin"
set "TMP=%TEMP%\orangu-install-%RANDOM%%RANDOM%"

where powershell >nul 2>&1 || (echo error: PowerShell is required & exit /b 1)

echo Fetching latest release...
for /f "usebackq delims=" %%v in (`powershell -NoProfile -NonInteractive -Command ^
    "(Invoke-RestMethod 'https://api.github.com/repos/%REPO%/releases/latest').tag_name"`) do set "VERSION=%%v"
if "!VERSION!"=="" (echo error: could not fetch latest release & exit /b 1)
echo Version: !VERSION!

set "ASSET=orangu-!VERSION!-x86_64-pc-windows-msvc.zip"
set "URL=https://github.com/%REPO%/releases/download/!VERSION!/!ASSET!"

echo Downloading !ASSET!...
mkdir "!TMP!" >nul 2>&1
powershell -NoProfile -NonInteractive -Command ^
    "Invoke-WebRequest -Uri '!URL!' -OutFile '!TMP!\!ASSET!' -UseBasicParsing" || ^
    (echo error: download failed & rd /s /q "!TMP!" >nul 2>&1 & exit /b 1)

powershell -NoProfile -NonInteractive -Command ^
    "$ErrorActionPreference='Stop'; Expand-Archive -Path '!TMP!\!ASSET!' -DestinationPath '!TMP!\out' -Force" || ^
    (echo error: could not extract archive & rd /s /q "!TMP!" >nul 2>&1 & exit /b 1)
if not exist "!TMP!\out\orangu.exe" (echo error: binary not found in archive & rd /s /q "!TMP!" >nul 2>&1 & exit /b 1)

if not exist "!INSTALL_DIR!" mkdir "!INSTALL_DIR!"
copy /y "!TMP!\out\orangu.exe" "!INSTALL_DIR!\orangu.exe" >nul || ^
    (echo error: could not write to !INSTALL_DIR! & exit /b 1)

rd /s /q "!TMP!" >nul 2>&1
echo Installed: !INSTALL_DIR!\orangu.exe
echo Run "orangu --help" to get started.
echo Run "orangu -s" and add the output to your PowerShell $PROFILE for completions.
endlocal

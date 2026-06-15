#!/bin/sh
# Copyright (C) 2026 The orangu community
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
# Usage: curl -fsSL https://raw.githubusercontent.com/mnemosyne-systems/orangu/main/install.sh | sh
# Override install directory: INSTALL_DIR=/usr/local/bin sh install.sh
set -e

REPO="mnemosyne-systems/orangu"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/bin}"

if command -v curl >/dev/null 2>&1; then
    fetch()    { curl -fsSL "$1"; }
    fetch_to() { curl -fsSL -o "$2" "$1"; }
elif command -v wget >/dev/null 2>&1; then
    fetch()    { wget -qO- "$1"; }
    fetch_to() { wget -qO "$2" "$1"; }
else
    echo "error: curl or wget is required" >&2; exit 1
fi

command -v tar >/dev/null 2>&1 || { echo "error: tar is required" >&2; exit 1; }

# Detect OS and architecture
case "$(uname -s)" in
    Linux)  OS="linux" ;;
    Darwin) OS="darwin" ;;
    *)      echo "error: unsupported OS: $(uname -s)" >&2; exit 1 ;;
esac

case "$(uname -m)" in
    x86_64)        ARCH="x86_64" ;;
    aarch64|arm64) ARCH="aarch64" ;;
    *)             echo "error: unsupported architecture: $(uname -m)" >&2; exit 1 ;;
esac

# On Linux, detect glibc vs musl
if [ "$OS" = "linux" ]; then
    if ldd --version 2>&1 | grep -qi musl; then LIBC="musl"; else LIBC="gnu"; fi
    TARGET="${ARCH}-unknown-linux-${LIBC}"
else
    TARGET="${ARCH}-apple-darwin"
fi

echo "Platform: ${TARGET}"

# Resolve latest release
VERSION=$(fetch "https://api.github.com/repos/${REPO}/releases/latest" \
    | grep '"tag_name"' | sed 's/.*"tag_name": *"\([^"]*\)".*/\1/')
[ -n "$VERSION" ] || { echo "error: could not fetch latest release" >&2; exit 1; }
echo "Version:  ${VERSION}"

ASSET="orangu-${VERSION}-${TARGET}.tar.gz"
URL="https://github.com/${REPO}/releases/download/${VERSION}/${ASSET}"

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "Downloading ${ASSET}..."
fetch_to "$URL" "${TMP}/${ASSET}" || { echo "error: download failed: ${URL}" >&2; exit 1; }

tar -xzf "${TMP}/${ASSET}" -C "$TMP" || { echo "error: could not extract ${ASSET}" >&2; exit 1; }
[ -f "${TMP}/orangu" ] || { echo "error: binary not found in archive" >&2; exit 1; }

mkdir -p "$INSTALL_DIR"
[ -w "$INSTALL_DIR" ] || { echo "error: ${INSTALL_DIR} is not writable — try sudo or set INSTALL_DIR" >&2; exit 1; }

cp "${TMP}/orangu" "${INSTALL_DIR}/orangu"
chmod +x "${INSTALL_DIR}/orangu"
echo "Installed: ${INSTALL_DIR}/orangu"

case ":${PATH}:" in
    *":${INSTALL_DIR}:"*) ;;
    *) echo "warning: ${INSTALL_DIR} is not in your PATH — add: export PATH=\"${INSTALL_DIR}:\$PATH\"" ;;
esac

echo ""
echo "Run 'orangu --help' to get started."
echo "Run 'orangu -s' to set up shell completions."

#!/bin/bash
set -euo pipefail

# Builds the documentation into target/doc:
#
#   manual      the user manual from doc/manual/en, as PDF and HTML
#   cheatsheet  the four-page cheat sheet from doc/cheatsheet/en, as PDF
#
# Both PDFs are drawn by orangu itself (`--build-manual`, `--build-cheatsheet`),
# with the same printpdf engine that writes the reports `/export` produces.
# Pandoc is used for the HTML manual alone.
#
# With no argument both are built.

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
readonly OUTPUT_DIR="$PROJECT_ROOT/target/doc"

readonly MANUAL_DIR="$SCRIPT_DIR/manual/en"
readonly MANUAL_PDF="$OUTPUT_DIR/orangu-en.pdf"
readonly MANUAL_HTML="$OUTPUT_DIR/orangu-en.html"
readonly RESOURCE_PATH="$SCRIPT_DIR:$MANUAL_DIR:$SCRIPT_DIR/manual"

readonly CHEATSHEET_DIR="$SCRIPT_DIR/cheatsheet/en"
readonly CHEATSHEET_PDF="$OUTPUT_DIR/orangu-cheatsheet-en.pdf"

usage() {
    echo "Usage: ${BASH_SOURCE[0]##*/} [manual | cheatsheet]" >&2
    exit 1
}

case "${1:-all}" in
    all)        build_manual=yes; build_cheatsheet=yes ;;
    manual)     build_manual=yes; build_cheatsheet=no ;;
    cheatsheet) build_manual=no;  build_cheatsheet=yes ;;
    *)          usage ;;
esac
[[ $# -le 1 ]] || usage

# Both PDFs are drawn by orangu, so the only thing pandoc is still needed for
# is the HTML manual.
if [[ "$build_manual" == yes ]] && ! command -v pandoc >/dev/null 2>&1; then
    echo "Error: pandoc is required for the HTML manual but was not found in PATH." >&2
    exit 1
fi

if ! command -v cargo >/dev/null 2>&1; then
    echo "Error: cargo is required: orangu draws its own PDFs." >&2
    exit 1
fi

# The sources of a document, in order: ??-*.md, one chapter — or, for the cheat
# sheet, one page — per file.
sources_in() {
    local dir="$1" found
    shopt -s nullglob
    found=("$dir"/??-*.md)
    shopt -u nullglob
    if [[ ${#found[@]} -eq 0 ]]; then
        echo "Error: no sources found in $dir matching ??-*.md" >&2
        exit 1
    fi
    printf '%s\n' "${found[@]}"
}

manual() {
    # Drawn by orangu itself, like the cheat sheet: one printpdf engine for
    # every PDF the project produces. The HTML manual is still pandoc's, since
    # a PDF engine cannot make one.
    echo "Generating PDF manual: $MANUAL_PDF"
    cargo run --quiet --bin orangu -- --build-manual "$MANUAL_DIR" "$MANUAL_PDF" --quiet

    local sources
    mapfile -t sources < <(sources_in "$MANUAL_DIR")

    echo "Generating HTML manual: $MANUAL_HTML"
    (
      cd "$SCRIPT_DIR"
      pandoc \
        -o "$MANUAL_HTML" \
        -s \
        --embed-resources \
        -f markdown-smart \
        --resource-path="$RESOURCE_PATH" \
        --css "$SCRIPT_DIR/manual/manual.css" \
        -N \
        --toc \
        -t html5 \
        "${sources[@]}"
    )
}

cheatsheet() {
    # Built by orangu itself: the same Rust engine that draws the PDFs
    # `/export` writes, so the card and the reports share their branding
    # without a second toolchain to keep in step. One source file per page,
    # and the build fails if a page's boxes outgrow it.
    echo "Generating PDF cheat sheet: $CHEATSHEET_PDF"
    cargo run --quiet --bin orangu -- \
        --build-cheatsheet "$CHEATSHEET_DIR" "$CHEATSHEET_PDF" --quiet
}

mkdir -p "$OUTPUT_DIR"

if [[ "$build_manual" == yes ]]; then
    manual
fi
if [[ "$build_cheatsheet" == yes ]]; then
    cheatsheet
fi

echo "Documentation generated:"
if [[ "$build_manual" == yes ]]; then
    echo "  $MANUAL_PDF"
    echo "  $MANUAL_HTML"
fi
if [[ "$build_cheatsheet" == yes ]]; then
    echo "  $CHEATSHEET_PDF"
fi

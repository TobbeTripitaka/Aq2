#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# compile_docs.sh  —  Compile LaTeX figures to PDF
#
# Usage:
#   ./compile_docs.sh                  # compile all documents
#   ./compile_docs.sh observables      # compile fig_observables only
#
# Run from project root. PDFs are written to output/documents/.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT_DIR="$(pwd)"
OUT_DIR="output/documents"
FIG_DIR="fig"

mkdir -p "$OUT_DIR"

# ── Helper: compile one tex file ──────────────────────────────────────────
# Usage: compile_tex <tex_file_relative_to_fig> <output_pdf_name>
compile_tex() {
    local TEX_FILE="$1"
    local OUT_NAME="$2"
    local TEX_BASE="${TEX_FILE%.tex}"

    echo "──────────────────────────────────────────────"
    echo "→ Compiling: fig/$TEX_FILE"

    cd "$ROOT_DIR/$FIG_DIR"

    # Two passes: first builds aux/toc, second resolves refs
    pdflatex -interaction=nonstopmode "$TEX_FILE" || true
    # Run bibtex if a .bib file exists alongside the tex
    if ls "${TEX_BASE}"*.bib 2>/dev/null | grep -q .; then
        bibtex "$TEX_BASE" || true
        pdflatex -interaction=nonstopmode "$TEX_FILE" || true
    fi
    pdflatex -interaction=nonstopmode "$TEX_FILE" || true

    cd "$ROOT_DIR"

    local PDF_SRC="$FIG_DIR/${TEX_BASE}.pdf"
    if [ -f "$PDF_SRC" ]; then
        mv "$PDF_SRC" "$OUT_DIR/$OUT_NAME"
        echo "✓ $OUT_DIR/$OUT_NAME"
    else
        echo "✗ PDF not found: $PDF_SRC" >&2
    fi
}

# ── Document registry ─────────────────────────────────────────────────────
# Add new documents here as: compile_tex <tex_file> <output_name>
compile_observables() {
    compile_tex "fig_observables.tex" "observables.pdf"
}

# ── Dispatch ──────────────────────────────────────────────────────────────
TARGET="${1:-all}"

case "$TARGET" in
    all | observables)
        compile_observables
        ;;
    *)
        echo "Unknown target: $TARGET"
        echo "Available targets: all, observables"
        exit 1
        ;;
esac

echo ""
echo "Done. Output in $ROOT_DIR/$OUT_DIR/"

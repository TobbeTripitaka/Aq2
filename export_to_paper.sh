#!/usr/bin/env bash
#
# export_to_paper.sh
#
# Usage:
#   ./export_to_paper.sh
#
# Description:
#   Run from the project root.
# This lines up all teh figures needed for the paper. 
#
set -uo pipefail

# --- Paths -------------------------------------------------------------
EXPORT_DIR="export/to_paper"
FIG_DIR="${EXPORT_DIR}/fig"
TEX_DIR="${EXPORT_DIR}/tex"
ZIP_PATH="export/to_paper.zip"
SRC_TEX_DIR="output/tex"

# --- Helpers -------------------------------------------------------------
warn() {
    echo "WARNING: $*" >&2
}

# Copy src -> dst, warning (but not failing) if src is missing.
safe_copy() {
    local src="$1"
    local dst="$2"
    if [[ -f "${src}" ]]; then
        cp -f -- "${src}" "${dst}"
    else
        warn "source file not found, skipping: ${src}"
    fi
}

# --- Ensure destination directories exist (without deleting existing) ---
mkdir -p -- "${FIG_DIR}" "${TEX_DIR}"

# --- Copy/rename figures -------------------------------------------------
safe_copy "fig/IHFC_q_map.png"                                "${FIG_DIR}/fig1a.pdf"
safe_copy "fig/IHFC_q_distribution_cleaning_steps.pdf"        "${FIG_DIR}/fig2a.pdf"
safe_copy "fig/IHFC_map.pdf"                                  "${FIG_DIR}/fig2b.pdf"
safe_copy "fig/5_TARGETS/ant_gbm_q50_corr_contour.png"        "${FIG_DIR}/fig3a.png"
safe_copy "fig/5_TARGETS/ant_qrf_q50_corr_contour.png"        "${FIG_DIR}/fig3b.png"
safe_copy "fig/5_TARGETS/ant_sim_q50_corr_contour.png"        "${FIG_DIR}/fig3c.png"
safe_copy "fig/5_TARGETS/grl_gbm_q50_corr_contour.png"        "${FIG_DIR}/fig4a.png"
safe_copy "fig/5_TARGETS/grl_qrf_q50_corr_contour.png"        "${FIG_DIR}/fig4b.png"
safe_copy "fig/5_TARGETS/grl_sim_q50_corr_contour.png"        "${FIG_DIR}/fig4c.png"
safe_copy "fig/7_ENSEMBLE/ant_ens_explainability.png"         "${FIG_DIR}/fig5a.png"
safe_copy "fig/7_ENSEMBLE/grl_ens_explainability.png"         "${FIG_DIR}/fig5b.png"

# --- Copy all .tex files from output/tex --------------------------------
if [[ -d "${SRC_TEX_DIR}" ]]; then
    shopt -s nullglob
    tex_files=("${SRC_TEX_DIR}"/*.tex)
    shopt -u nullglob

    if [[ ${#tex_files[@]} -eq 0 ]]; then
        warn "no .tex files found in ${SRC_TEX_DIR}"
    else
        for tex_file in "${tex_files[@]}"; do
            cp -f -- "${tex_file}" "${TEX_DIR}/"
        done
    fi
else
    warn "source directory not found, skipping: ${SRC_TEX_DIR}"
fi

# --- Create zip archive for Overleaf upload ------------------------------
# Build the archive from within export/to_paper so the zip contains the
# fig/ and tex/ folders at its root (suitable for direct Overleaf upload).
if command -v zip >/dev/null 2>&1; then
    rm -f -- "${ZIP_PATH}"
    (cd "${EXPORT_DIR}" && zip -r -q "../../${ZIP_PATH}" .)
    echo "Created archive: ${ZIP_PATH}"
else
    warn "'zip' command not found; skipping archive creation"
fi

echo "Export complete: ${EXPORT_DIR}"

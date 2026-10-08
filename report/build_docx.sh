#!/usr/bin/env bash
# Build report/TECHNICAL_REPORT.docx and the PDF that follows from it.
#
# The Word path exists because a reviewer's export read better than the LaTeX
# build -- filled table headers, striped rows, tighter section headings -- but
# that export had been edited by hand and had drifted from the audited Markdown.
# So the design was kept and the fork was not: neuromps.report.docx renders the
# same source build_report.sh uses, and LibreOffice converts the result. The PDF
# is therefore a rendering of the .docx, and the two cannot disagree.
#
# Usage:
#   report/build_docx.sh                       # .docx and .pdf for the technical report
#   report/build_docx.sh --docx-only           # skip the PDF conversion
#   report/build_docx.sh --check-pages 15 20   # also gate the main-matter length
#   report/build_docx.sh --source report/KAGGLE_WRITEUP.md
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="${REPO_ROOT}/report/TECHNICAL_REPORT.md"
TEMPLATE="${REPO_ROOT}/report/reference.docx"
docx_only=""
min_pages=""
max_pages=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) SOURCE="${2:?--source needs a path}"; shift 2 ;;
    --template) TEMPLATE="${2:?--template needs a path}"; shift 2 ;;
    --docx-only) docx_only="yes"; shift ;;
    --check-pages)
      min_pages="${2:?--check-pages needs MIN and MAX}"; max_pages="${3:?--check-pages needs MIN and MAX}"; shift 3 ;;
    -h|--help) sed -n '2,13p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

die() { echo "build_docx: $*" >&2; exit 1; }

[[ -f "${SOURCE}" ]] || die "missing source: ${SOURCE}"
[[ -f "${TEMPLATE}" ]] || die "missing style template: ${TEMPLATE}"

PYTHON="${PYTHON:-$(test -x "${REPO_ROOT}/.venv/bin/python" && echo "${REPO_ROOT}/.venv/bin/python" || command -v python3)}"
DOCX="${SOURCE%.md}.docx"
PDF="${SOURCE%.md}.pdf"

PYTHONPATH="${REPO_ROOT}/src" "${PYTHON}" -m neuromps.report.docx \
  --source "${SOURCE}" --template "${TEMPLATE}" --output "${DOCX}" \
  || die "the Word render failed"

[[ -n "${docx_only}" ]] && exit 0

command -v soffice >/dev/null || die "soffice is required to convert the .docx (apt install libreoffice-writer)"

# LibreOffice writes into the user profile on first run, and a headless run that
# shares the desktop profile fails when one is already open. Give it its own.
PROFILE="$(mktemp -d)"
trap 'rm -rf "${PROFILE}"' EXIT
soffice --headless --norestore \
  "-env:UserInstallation=file://${PROFILE}" \
  --convert-to pdf --outdir "$(dirname "${PDF}")" "${DOCX}" >/dev/null \
  || die "LibreOffice failed to convert ${DOCX}"

[[ -s "${PDF}" ]] || die "no PDF produced at ${PDF}"

# A conversion that silently drops the images is the failure worth catching: it
# is exactly what happened to the export this design came from.
if command -v pdfimages >/dev/null; then
  embedded="$(pdfimages -list "${PDF}" | tail -n +3 | wc -l)"
  wanted="$(grep -cE '^!\[' "${SOURCE}")"
  (( embedded >= wanted )) || die "the PDF holds ${embedded} images, the source has ${wanted} figures"
  echo "build_docx: ${embedded} image objects for ${wanted} figures"
fi

pages="$(pdfinfo "${PDF}" 2>/dev/null | awk '/^Pages:/ {print $2}')"
echo "build_docx: wrote ${DOCX} and ${PDF} - ${pages:-?} pages, $(du -h "${PDF}" | cut -f1)"

# The competition asks for 15-20 pages "excluding references and appendices", so
# the gate counts main matter only. Scanned from the back, because the contents
# table lists the appendices too and a forward scan would stop on it. The Word
# design sets tighter than the LaTeX one and first landed at 14 pages, under the
# floor -- which is the whole reason this is a check and not a glance.
count_main_matter() {
  local page
  for (( page = pages; page >= 1; page-- )); do
    if pdftotext -f "${page}" -l "${page}" "${PDF}" - 2>/dev/null | grep -qE '^[[:space:]]*Appendix A\b'; then
      echo $(( page - 1 )); return 0
    fi
  done
  echo "${pages}"
}

if [[ -n "${min_pages}" ]]; then
  command -v pdftotext >/dev/null || die "--check-pages needs pdftotext (apt install poppler-utils)"
  main="$(count_main_matter)"
  (( main >= min_pages && main <= max_pages )) \
    || die "main matter is ${main} pages, outside the required range ${min_pages}-${max_pages}"
  echo "build_docx: main matter ${main} pages is within ${min_pages}-${max_pages}"
fi

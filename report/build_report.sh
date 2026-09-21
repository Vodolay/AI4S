#!/usr/bin/env bash
# Build report/TECHNICAL_REPORT.pdf from the Markdown source.
#
# Primary path is pandoc with xelatex. If xelatex is missing, the script falls
# back to pandoc -> HTML -> weasyprint, which produces a similar but not
# identical document; the fallback is announced rather than silent, because a
# page-count check downstream should know which engine produced the file.
#
# Title-page fields that can go stale are inserted at build time from the
# repository itself: the commit, whether the tree was dirty, the remote URL and
# the build date. Team and category come from the environment so they are set in
# one place rather than copied into the prose.
#
# Usage:
#   report/build_report.sh                    # build the PDF
#   report/build_report.sh --check-pages 14 22   # build, then require a page range
#   report/build_report.sh --engine html      # force the weasyprint fallback
#   report/build_report.sh --dry-run          # show resolved metadata, build nothing
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="${REPO_ROOT}/report/TECHNICAL_REPORT.md"
OUTPUT="${REPO_ROOT}/report/TECHNICAL_REPORT.pdf"
CSS="${REPO_ROOT}/report/report.css"

TEAM="${TEAM:-NeuroMPS}"
CATEGORY="${CATEGORY:-AI4S Open Innovation: AI + Organ-on-a-Chip (5th Pazhou Algorithm Competition)}"

engine="auto"
check_pages=""
min_pages=""
max_pages=""
dry_run=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check-pages)
      check_pages="yes"; min_pages="${2:?--check-pages needs MIN and MAX}"; max_pages="${3:?--check-pages needs MIN and MAX}"; shift 3 ;;
    --engine) engine="${2:?--engine needs xelatex or html}"; shift 2 ;;
    --output) OUTPUT="${2:?--output needs a path}"; shift 2 ;;
    --dry-run) dry_run="yes"; shift ;;
    -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

die() { echo "build_report: $*" >&2; exit 1; }

# Argument errors are reported before anything about the environment: a typo in
# --engine should say so whether or not pandoc happens to be installed.
case "${engine}" in
  auto|xelatex|html) ;;
  *) die "unknown engine: ${engine} (expected auto, xelatex or html)" ;;
esac
if [[ -n "${check_pages}" ]]; then
  [[ "${min_pages}" =~ ^[0-9]+$ && "${max_pages}" =~ ^[0-9]+$ ]] \
    || die "--check-pages needs two integers, got '${min_pages}' and '${max_pages}'"
  (( min_pages <= max_pages )) || die "--check-pages MIN must not exceed MAX"
fi

# --- metadata resolved from the repository -----------------------------------

git_value() { git -C "${REPO_ROOT}" "$@" 2>/dev/null || true; }

COMMIT="$(git_value rev-parse --short HEAD)"
COMMIT="${COMMIT:-unknown}"
if [[ -n "$(git_value status --porcelain)" ]]; then
  COMMIT="${COMMIT} (working tree dirty)"
fi

REPO_URL="$(git_value remote get-url origin)"
case "${REPO_URL}" in
  git@*:*) REPO_URL="https://$(echo "${REPO_URL}" | sed -e 's/^git@//' -e 's/:/\//' -e 's/\.git$//')" ;;
  *.git)   REPO_URL="${REPO_URL%.git}" ;;
  "")      REPO_URL="no remote configured" ;;
esac

BUILD_DATE="$(date -u '+%Y-%m-%d')"

if [[ -n "${dry_run}" ]]; then
  printf 'team:       %s\ncategory:   %s\nrepository: %s\ncommit:     %s\nbuilt:      %s\nsource:     %s\noutput:     %s\n' \
    "${TEAM}" "${CATEGORY}" "${REPO_URL}" "${COMMIT}" "${BUILD_DATE}" "${SOURCE}" "${OUTPUT}"
  exit 0
fi

[[ -f "${SOURCE}" ]] || die "missing source: ${SOURCE}"
command -v pandoc >/dev/null || die "pandoc is required (apt install pandoc, or see report/README build notes)"

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

# --- title page --------------------------------------------------------------

cat > "${WORK}/titlepage.tex" <<LATEX
\\begin{titlepage}
\\centering
\\vspace*{3cm}
{\\Huge\\bfseries NeuroMPS\\par}
\\vspace{0.6cm}
{\\large A provenance-tracked MEA data asset and a calibrated two-compartment digital twin\\par}
\\vspace{2.5cm}
\\begin{tabular}{rl}
\\textbf{Team} & ${TEAM}\\\\[4pt]
\\textbf{Category} & \\begin{minipage}[t]{9cm}${CATEGORY}\\end{minipage}\\\\[10pt]
\\textbf{Repository} & \\texttt{${REPO_URL}}\\\\[4pt]
\\textbf{Commit} & \\texttt{${COMMIT}}\\\\[4pt]
\\textbf{Built} & ${BUILD_DATE}\\\\
\\end{tabular}
\\vfill
{\\small Research decision-support only. No efficacy, safety, diagnostic or clinical claim is made.\\par}
\\end{titlepage}
LATEX

cat > "${WORK}/titlepage.html" <<HTML
<div class="titlepage">
  <h1 class="title">NeuroMPS</h1>
  <p class="subtitle">A provenance-tracked MEA data asset and a calibrated two-compartment digital twin</p>
  <table class="titlemeta">
    <tr><th>Team</th><td>${TEAM}</td></tr>
    <tr><th>Category</th><td>${CATEGORY}</td></tr>
    <tr><th>Repository</th><td><code>${REPO_URL}</code></td></tr>
    <tr><th>Commit</th><td><code>${COMMIT}</code></td></tr>
    <tr><th>Built</th><td>${BUILD_DATE}</td></tr>
  </table>
  <p class="disclaimer">Research decision-support only. No efficacy, safety, diagnostic or
  clinical claim is made.</p>
</div>
HTML

# The Markdown starts with its own H1 and disclaimer, which the title page now
# carries; strip them so the document opens on the table of contents.
sed -e '1,8{/^# NeuroMPS/d; /^\*\*Technical report\.\*\*/d; /^(5th Pazhou/d; /^> \*\*Research decision-support only/d; /^> clinical claim/d;}' \
  "${SOURCE}" > "${WORK}/body.md"

# --- build -------------------------------------------------------------------

# The report uses Greek, arrows and comparison signs. Latin Modern, which pandoc
# selects by default under xelatex, does not cover them in text mode, and
# xelatex drops a missing glyph with a warning rather than an error - so the
# characters would disappear from the PDF silently. Pick a font that has them.
pick_font() {
  local kind="$1" candidate
  command -v fc-list >/dev/null || return 1
  if [[ "${kind}" == "mono" ]]; then
    for candidate in "DejaVu Sans Mono" "Liberation Mono" "Noto Sans Mono"; do
      fc-list : family | grep -qiF "${candidate}" && { echo "${candidate}"; return 0; }
    done
  else
    for candidate in "DejaVu Serif" "Liberation Serif" "Noto Serif" "TeX Gyre Termes"; do
      fc-list : family | grep -qiF "${candidate}" && { echo "${candidate}"; return 0; }
    done
  fi
  return 1
}

build_xelatex() {
  command -v xelatex >/dev/null || return 1
  local font_args=()
  local mainfont monofont
  if mainfont="$(pick_font main)"; then
    font_args+=(-V "mainfont=${mainfont}")
    echo "build_report: using ${mainfont} for text coverage of Greek and arrows" >&2
  else
    echo "build_report: no wide-coverage font found; Greek and arrows may be dropped" >&2
  fi
  if monofont="$(pick_font mono)"; then
    font_args+=(-V "monofont=${monofont}")
  fi
  pandoc "${WORK}/body.md" \
    --from=markdown+tex_math_dollars+pipe_tables+implicit_figures \
    --pdf-engine=xelatex \
    --resource-path="${REPO_ROOT}/report:${REPO_ROOT}" \
    --include-before-body="${WORK}/titlepage.tex" \
    --toc --toc-depth=2 \
    --number-sections \
    --highlight-style=tango \
    -V documentclass=article \
    -V fontsize=11pt \
    -V geometry:margin=2.5cm \
    -V linkcolor=black -V urlcolor=black -V toccolor=black \
    -V colorlinks=true \
    -V papersize=a4 \
    -V graphics=true \
    "${font_args[@]}" \
    -M date="${BUILD_DATE}" \
    -o "${OUTPUT}"
}

build_html() {
  command -v weasyprint >/dev/null || return 1
  [[ -f "${CSS}" ]] || die "missing ${CSS} for the HTML fallback"
  pandoc "${WORK}/body.md" \
    --from=markdown+pipe_tables+implicit_figures \
    --to=html5 --standalone \
    --resource-path="${REPO_ROOT}/report:${REPO_ROOT}" \
    --include-before-body="${WORK}/titlepage.html" \
    --toc --toc-depth=2 \
    --number-sections \
    --metadata title="NeuroMPS technical report" \
    -o "${WORK}/report.html"
  weasyprint --base-url "${REPO_ROOT}/report/" -s "${CSS}" "${WORK}/report.html" "${OUTPUT}"
}

used=""
case "${engine}" in
  xelatex) build_xelatex && used="xelatex" || die "xelatex path failed or xelatex is not installed" ;;
  html)    build_html && used="weasyprint" || die "html path failed or weasyprint is not installed" ;;
  auto)
    if build_xelatex; then
      used="xelatex"
    else
      echo "build_report: xelatex unavailable, falling back to pandoc -> HTML -> weasyprint" >&2
      build_html && used="weasyprint" || die "neither xelatex nor weasyprint is available"
    fi ;;
esac

[[ -s "${OUTPUT}" ]] || die "no output produced at ${OUTPUT}"

# --- page count --------------------------------------------------------------

count_pages() {
  local pdf="$1"
  if command -v pdfinfo >/dev/null; then
    pdfinfo "${pdf}" | awk '/^Pages:/ {print $2; found=1} END {exit !found}' && return 0
  fi
  python3 - "${pdf}" <<'PYTHON'
import re
import sys

data = open(sys.argv[1], "rb").read()
counts = [int(n) for n in re.findall(rb"/Type\s*/Pages[^>]*?/Count\s+(\d+)", data, re.S)]
pages = max(counts) if counts else len(re.findall(rb"/Type\s*/Page[^s]", data))
if not pages:
    sys.exit("could not determine the page count")
print(pages)
PYTHON
}

# A PDF that built cleanly can still have lost characters the font did not
# cover, so check the text layer for the ones most at risk.
verify_text() {
  command -v pdftotext >/dev/null || {
    echo "build_report: pdftotext absent, skipping the text check" >&2
    return 0
  }
  local text missing=() marker expanded
  text="$(pdftotext "${OUTPUT}" - 2>/dev/null)" || return 0
  # "NeuroMPS" appears only on the generated title page, since the Markdown H1 is
  # stripped above - so it doubles as a check that the title page was included.
  # The last two are the characters most likely to be dropped by a font without
  # Greek or mathematical coverage.
  for marker in "NeuroMPS" "Directionality" "Limitations" '\u03c1' '\u2265'; do
    expanded="$(printf '%b' "${marker}")"
    grep -qF -- "${expanded}" <<<"${text}" || missing+=("${expanded}")
  done
  if (( ${#missing[@]} )); then
    die "the PDF text layer is missing: ${missing[*]} (a font may lack these glyphs)"
  fi
  echo "build_report: text layer contains the expected Greek and comparison characters"
}

verify_text

pages="$(count_pages "${OUTPUT}")"
size="$(du -h "${OUTPUT}" | cut -f1)"
echo "build_report: wrote ${OUTPUT} via ${used} - ${pages} pages, ${size}"

if [[ -n "${check_pages}" ]]; then
  if (( pages < min_pages || pages > max_pages )); then
    die "page count ${pages} is outside the required range ${min_pages}-${max_pages}"
  fi
  echo "build_report: page count ${pages} is within ${min_pages}-${max_pages}"
fi

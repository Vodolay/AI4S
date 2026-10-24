"""Checks on the PDF build that do not need pandoc installed.

The build itself runs in CI, where pandoc and xelatex are available. These tests
guard the wiring around it: that the script is syntactically valid and resolves
its metadata, that every figure path in the report exists, and that the page
range required by the submission is stated identically in the Makefile and the
workflow, so the two cannot drift apart.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "report" / "build_report.sh"
REPORT = REPO_ROOT / "report" / "TECHNICAL_REPORT.md"
MAKEFILE = REPO_ROOT / "Makefile"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
# The competition asks for 15-20 pages "excluding references and appendices", so
# the gate counts main matter only: `count_main_matter` in the build script finds
# the page the appendices start on.
# Round 1 capped the main matter at 20 pages; the final round has no limit,
# so only the minimum is gated (it still catches a build that drops sections).
PAGES_MIN = 15


def test_the_build_script_exists_and_is_executable() -> None:
    assert SCRIPT.exists()
    assert os.access(SCRIPT, os.X_OK), "build_report.sh must be committed executable"


def test_the_build_script_is_valid_bash() -> None:
    result = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_dry_run_resolves_the_title_page_fields_from_the_repository() -> None:
    """Commit, repository and date are inserted at build time, never typed in."""
    result = subprocess.run(
        [str(SCRIPT), "--dry-run"], capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert result.returncode == 0, result.stderr
    output = result.stdout
    for field in ("team:", "category:", "repository:", "commit:", "built:"):
        assert field in output, output
    commit = re.search(r"commit:\s+([0-9a-f]{7,40})", output)
    assert commit, output
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT
    ).stdout.strip()
    assert commit.group(1) == head


def test_dry_run_honours_an_overridden_team_and_category() -> None:
    environment = {**os.environ, "TEAM": "Some Team", "CATEGORY": "Some Category"}
    result = subprocess.run(
        [str(SCRIPT), "--dry-run"], capture_output=True, text=True, cwd=REPO_ROOT, env=environment
    )
    assert "Some Team" in result.stdout
    assert "Some Category" in result.stdout


def test_an_ssh_remote_is_rewritten_as_a_browsable_url() -> None:
    """A submission needs a URL a reviewer can open, not a git+ssh address."""
    result = subprocess.run(
        [str(SCRIPT), "--dry-run"], capture_output=True, text=True, cwd=REPO_ROOT
    )
    line = next(li for li in result.stdout.splitlines() if li.startswith("repository:"))
    assert "git@" not in line
    assert line.split(":", 1)[1].strip().startswith(("http", "no remote"))


def test_every_figure_referenced_by_the_report_exists() -> None:
    """A missing figure fails the PDF build; this says which one, and earlier."""
    missing = [
        path
        for path in re.findall(r"!\[[^\]]*\]\((\.\./[^)]+)\)", REPORT.read_text())
        if not (REPO_ROOT / "report" / path).resolve().exists()
    ]
    assert missing == []


def test_figures_carry_captions_that_say_whether_data_are_real_or_simulated() -> None:
    """The project rule for figures applies to the report's captions too."""
    captions = re.findall(r"!\[([^\]]+)\]\(\.\./[^)]+\)", REPORT.read_text())
    assert len(captions) >= 10
    for caption in captions:
        assert re.search(r"\bREAL\b|\bSIMULATED\b|\bSIMULATIONS\b", caption), caption


def test_the_page_minimum_is_the_same_in_the_makefile_and_the_workflow() -> None:
    """Two places state the requirement; they must not drift, and neither caps the length."""
    makefile = MAKEFILE.read_text()
    assert f"PAGES_MIN ?= {PAGES_MIN}" in makefile
    assert "PAGES_MAX" not in makefile
    assert re.search(rf"--check-pages {PAGES_MIN}\s*$", WORKFLOW.read_text(), re.M)


def test_the_makefile_exposes_the_report_targets() -> None:
    makefile = MAKEFILE.read_text()
    for target in ("report:", "report-check:"):
        assert f"\n{target}" in makefile


def test_the_workflow_checks_the_pdf_that_is_submitted() -> None:
    """The page range is a property of the submitted PDF, which is the Word build.

    CI used to gate the LaTeX build. Once the report carried its model diagrams
    the two disagreed by four pages - the LaTeX one failing at 22 while the
    submitted one sat at 18 - so the gate was enforcing a limit on a document
    nobody submits.
    """
    workflow = WORKFLOW.read_text()
    assert "report/build_docx.sh --check-pages" in workflow
    assert "build_report.sh --check-pages" not in workflow
    assert "libreoffice-writer-nogui" in workflow
    # Without Carlito, LibreOffice substitutes a wider face for Calibri and every
    # table reflows, so CI would count different pages from the author's machine.
    assert "fonts-crosextra-carlito" in workflow
    assert "upload-artifact" in workflow


def test_the_workflow_installs_the_renderer_at_the_pinned_version() -> None:
    """The report job installs python-docx alone; the pin must not drift from pyproject."""
    pinned = re.search(r'"python-docx==([\d.]+)"', (REPO_ROOT / "pyproject.toml").read_text())
    assert pinned, "python-docx is not pinned in pyproject.toml"
    assert f"python-docx=={pinned.group(1)}" in WORKFLOW.read_text()


def test_the_fallback_stylesheet_matches_the_latex_geometry() -> None:
    """The two engines must produce comparable page counts, or the check is meaningless."""
    css = (REPO_ROOT / "report" / "report.css").read_text()
    assert "margin: 2.5cm" in css
    assert "font-size: 11pt" in css
    assert "counter-increment: figure" in css  # numbered captions, as in LaTeX


def test_the_script_rejects_an_unknown_engine() -> None:
    result = subprocess.run(
        [str(SCRIPT), "--engine", "nonsense"], capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert result.returncode != 0
    assert "nonsense" in (result.stderr + result.stdout)


@pytest.mark.skipif(shutil.which("pandoc") is None, reason="pandoc not installed")
def test_the_latex_path_still_renders_and_counts_main_matter(tmp_path: Path) -> None:
    """The LaTeX path still builds the companions, so it must still render and count.

    It no longer gates the technical report's length: its wider margins and 11 pt
    serif set the same text several pages longer than the submitted Word build,
    so its count says nothing about the submission.
    """
    output = tmp_path / "report.pdf"
    result = subprocess.run(
        [str(SCRIPT), "--output", str(output)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert output.exists() and output.stat().st_size > 50_000
    # The counter must separate main matter from appendices, because the
    # companions built through this path are checked the same way.
    assert "pages of main matter" in result.stdout, result.stdout
    main, total = (
        int(n)
        for n in re.search(r"(\d+) pages of main matter, (\d+) including", result.stdout).groups()
    )
    assert main < total, "appendices should not count toward the page range"


def test_every_figure_in_the_report_is_tracked_by_git() -> None:
    """Figures are gitignored by habit; the report embeds them, so they must not be.

    Without this, a fresh clone - which is what CI builds from - has no figures
    and the PDF build fails on a missing image.
    """
    figures = re.findall(r"!\[[^\]]*\]\((\.\./[^)]+\.png)\)", REPORT.read_text())
    assert figures
    ignored = []
    for figure in figures:
        path = (REPO_ROOT / "report" / figure).resolve().relative_to(REPO_ROOT)
        result = subprocess.run(
            ["git", "check-ignore", "-q", str(path)], cwd=REPO_ROOT, capture_output=True
        )
        if result.returncode == 0:
            ignored.append(str(path))
    assert ignored == [], f"figures the report needs are gitignored: {ignored}"


def test_the_build_refuses_stale_figures_when_asked() -> None:
    """The guard against a quick run leaving figures that disagree with metrics."""
    help_text = SCRIPT.read_text()
    assert "--strict-figures" in help_text
    assert "predate outputs/metrics.json" in help_text


@pytest.mark.skipif(shutil.which("pandoc") is None, reason="pandoc not installed")
def test_the_script_builds_a_source_other_than_the_technical_report(tmp_path: Path) -> None:
    """`--source` exists so a companion document gets the same typography.

    Without it each document grows its own half-correct build: the font
    selection, the metrics-comment stripping and the engine fallback here were
    all hard-won and should not be reimplemented per document.
    """
    source = tmp_path / "companion.md"
    source.write_text(
        "# A companion\n\n"
        "**Companion to `report/TECHNICAL_REPORT.md`.** Opening line that must survive.\n\n"
        "> **Research decision-support only.** Stripped, because the title page carries it.\n\n"
        "## A section\n\nBody text with a citation 0.537 "
        "<!-- metrics.json: some.key --> that renders without a gap.\n"
    )
    output = tmp_path / "companion.pdf"
    result = subprocess.run(
        [
            str(SCRIPT),
            "--source",
            str(source),
            "--output",
            str(output),
            "--title",
            "A companion",
            "--subtitle",
            "Subtitle here",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert output.exists()
    if shutil.which("pdftotext") is None:
        return
    text = subprocess.run(["pdftotext", str(output), "-"], capture_output=True, text=True).stdout
    assert "A companion" in text, "the title page should carry the given title"
    assert "Subtitle here" in text
    assert "Opening line that must survive" in text
    # The title page carries its own disclaimer footer, so the phrase is present;
    # what must be gone is the source's blockquote, identified by its own words.
    assert "Stripped, because the title page carries it" not in text
    assert "metrics.json" not in text, "citation comments are not content"


@pytest.mark.skipif(shutil.which("pandoc") is None, reason="pandoc not installed")
def test_source_defaults_its_output_beside_the_source(tmp_path: Path) -> None:
    source = tmp_path / "doc.md"
    source.write_text("# Doc\n\n## Section\n\nBody.\n")
    result = subprocess.run(
        [str(SCRIPT), "--source", str(source), "--title", "Doc", "--subtitle", "S"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "doc.pdf").exists()


def _report_documents() -> list[Path]:
    """Every report document, excluding external critiques."""
    external = {"REVIEW.md", "CONTEXT_FROM_CHAT.md"}
    return sorted(p for p in (REPO_ROOT / "report").glob("*.md") if p.name not in external)


def _malformed_figures(text: str) -> list[str]:
    """Image references that markdown will not render as images.

    `test_every_figure_referenced_by_the_report_exists` and its caption
    neighbour both match a complete `![alt](path)`, so a figure whose `![` is
    missing is invisible to the very checks meant to catch broken figures. One
    did reach the committed report that way: an edit moved a figure to an
    appendix and split its caption, leaving an unterminated `![` in the
    appendix and a stray `](path)` six hundred lines earlier. Both halves
    render as literal text in the PDF.

    An unterminated `![` is found by asking whether another `![` opens before
    this one closes, rather than by looking for a closing bracket at all: the
    stranded opener in the appendix would otherwise borrow the closing bracket
    of the next figure and look complete.
    """
    problems = []
    openers = [m.start() for m in re.finditer(r"!\[", text)]
    for position in openers:
        closing = re.search(r"\]\(", text[position:])
        following = next((o for o in openers if o > position), None)
        if closing is None or (following is not None and following < position + closing.start()):
            problems.append(f"unterminated '![' at line {text[:position].count(chr(10)) + 1}")
    for match in re.finditer(r"\]\((\.\./[^)]+\.png)\)", text):
        opener = text.rfind("![", 0, match.start())
        closer = text.rfind("](", 0, match.start())
        if opener == -1 or (closer != -1 and closer > opener):
            problems.append(f"no opening '![' for {match.group(1)}")
    return problems


@pytest.mark.parametrize("document", _report_documents(), ids=lambda p: p.name)
def test_no_report_has_a_figure_that_will_not_render(document: Path) -> None:
    assert _malformed_figures(document.read_text()) == [], document.name


def test_the_malformed_figure_check_catches_both_halves_of_a_split_caption() -> None:
    """The check is only worth having if it fails on the shape that got through.

    That shape is not a caption broken over a blank line - markdown renders
    that fine - but a caption whose two halves ended up in different sections
    with complete figures between them.
    """
    split = (
        "![Stranded opener, REAL data. The\n\n"
        "![A whole figure. REAL.](../outputs/figures/whole.png)\n\n"
        "tail of the stranded caption.](../outputs/figures/split.png){width=51%}\n"
    )
    problems = _malformed_figures(split)
    assert any("unterminated" in p for p in problems), problems
    assert any("no opening" in p and "split.png" in p for p in problems), problems
    assert _malformed_figures("![A whole caption. REAL.](../outputs/figures/x.png)\n") == []
    assert (
        _malformed_figures(
            "![A caption wrapped\nover two lines. REAL.](../outputs/figures/x.png)\n"
        )
        == []
    )


@pytest.mark.skipif(shutil.which("pandoc") is None, reason="pandoc not installed")
def test_a_long_front_matter_does_not_split_the_disclaimer(tmp_path: Path) -> None:
    """The disclaimer is stripped whole, however long the opening paragraph is.

    The strip used to be bounded to the first ten lines. A document whose opening
    paragraph ran long put the disclaimer's first line inside that window and its
    second line outside, so the built PDF opened with a sentence fragment -
    visible to a reader, invisible to every check.
    """
    source = tmp_path / "long_front_matter.md"
    filler = "\n".join(
        f"Opening paragraph line {i}, which exists to run the header long." for i in range(1, 9)
    )
    source.write_text(
        "# A title\n\n"
        f"**Companion to something.** {filler}\n\n"
        "> **Research decision-support only.** Nothing here is an efficacy or\n"
        "> clinical claim, and nothing here should be used to make one.\n\n"
        "## 1. A section\n\nBody text.\n"
    )
    output = tmp_path / "out.pdf"
    result = subprocess.run(
        [
            str(SCRIPT),
            "--source",
            str(source),
            "--output",
            str(output),
            "--title",
            "A title",
            "--subtitle",
            "A subtitle",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    if shutil.which("pdftotext") is None:
        return
    text = subprocess.run(["pdftotext", str(output), "-"], capture_output=True, text=True).stdout
    flat = " ".join(text.split())
    assert "should be used to make one" not in flat, (
        "the disclaimer's second line survived the strip and opens the document as a fragment"
    )


V2 = REPO_ROOT / "report" / "TECHNICAL_REPORT.md"


def test_the_v2_report_exists_and_is_tracked() -> None:
    """v2 is a submission document, not a gitignored companion."""
    assert V2.exists()
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", str(V2.relative_to(REPO_ROOT))],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    assert ignored.returncode != 0, "the submission report must not be gitignored"


def test_the_v2_report_declares_what_the_competition_asks_for() -> None:
    """Category, team and the AI-assistance disclosure are submission requirements."""
    text = V2.read_text()
    assert "Submission category:" in text
    assert "**Team.**" in text
    assert "AI assistance disclosure" in text
    assert "decision-support only" in text


def test_the_v2_report_carries_no_audience_specific_framing() -> None:
    """It was adapted from a companion written for one audience; that framing is gone."""
    text = V2.read_text().lower()
    for phrase in (
        "companion to",
        "for biologists",
        "wet-lab expertise",
        "explained for the bench",
    ):
        assert phrase not in text, f"audience framing survived: {phrase}"


@pytest.mark.skipif(shutil.which("soffice") is None, reason="LibreOffice not installed")
def test_the_submitted_report_builds_with_at_least_the_minimum_length(tmp_path: Path) -> None:
    """At least 15 pages of main matter, counted on the Word build that is submitted.

    Built into a temporary directory: the default writes beside the source, which
    would overwrite the committed .docx - possibly one just edited by hand.
    """
    committed = (REPO_ROOT / "report" / "TECHNICAL_REPORT.docx").read_bytes()
    result = subprocess.run(
        [
            str(REPO_ROOT / "report" / "build_docx.sh"),
            "--outdir",
            str(tmp_path),
            "--check-pages",
            str(PAGES_MIN),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    main = int(re.search(r"main matter (\d+) pages", result.stdout).group(1))
    total = int(re.search(r" - (\d+) pages,", result.stdout).group(1))
    assert main >= PAGES_MIN
    assert main < total, "appendices should not count toward the page range"
    assert (REPO_ROOT / "report" / "TECHNICAL_REPORT.docx").read_bytes() == committed, (
        "the test build overwrote the committed .docx"
    )

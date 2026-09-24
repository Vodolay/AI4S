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
PAGES_MIN, PAGES_MAX = 15, 20


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


def test_the_page_range_is_the_same_in_the_makefile_and_the_workflow() -> None:
    """Two places state the requirement; they must not drift."""
    makefile = MAKEFILE.read_text()
    assert f"PAGES_MIN ?= {PAGES_MIN}" in makefile
    assert f"PAGES_MAX ?= {PAGES_MAX}" in makefile
    assert f"--check-pages {PAGES_MIN} {PAGES_MAX}" in WORKFLOW.read_text()


def test_the_makefile_exposes_the_report_targets() -> None:
    makefile = MAKEFILE.read_text()
    for target in ("report:", "report-check:"):
        assert f"\n{target}" in makefile


def test_the_workflow_builds_the_pdf_and_keeps_it() -> None:
    workflow = WORKFLOW.read_text()
    assert "report/build_report.sh --check-pages" in workflow
    assert "texlive-xetex" in workflow
    assert "upload-artifact" in workflow


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
def test_the_pdf_builds_and_lands_in_the_required_page_range(tmp_path: Path) -> None:
    """The real build, when the toolchain is present (CI, and any dev box with pandoc)."""
    output = tmp_path / "report.pdf"
    result = subprocess.run(
        [str(SCRIPT), "--output", str(output), "--check-pages", str(PAGES_MIN), str(PAGES_MAX)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert output.exists() and output.stat().st_size > 50_000
    # The gate must be counting main matter, not the whole document: the report
    # has appendices, so the two numbers have to differ and the range has to be
    # applied to the smaller one.
    assert "pages of main matter" in result.stdout, result.stdout
    main, total = (
        int(n)
        for n in re.search(r"(\d+) pages of main matter, (\d+) including", result.stdout).groups()
    )
    assert main < total, "appendices should not count toward the page range"
    assert PAGES_MIN <= main <= PAGES_MAX


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

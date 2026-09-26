"""Every number cited in a report must match the metrics that produced it."""

import json
from pathlib import Path

import pytest

from neuromps.report.verify import (
    REPORT_DIR,
    check_documents,
    check_text,
    citation_count,
    compare,
    lookup,
)

# REVIEW.md is an external critique of these documents, not one of their claims:
# it quotes numbers in order to dispute them and names clinical vocabulary in
# order to report that the reports avoid it. Holding it to the project's own
# citation and language rules would be a category error.
EXTERNAL = {"REVIEW.md", "CONTEXT_FROM_CHAT.md"}
REPORTS = sorted(p for p in REPORT_DIR.glob("*.md") if p.name not in EXTERNAL)


@pytest.mark.parametrize("document", REPORTS, ids=lambda p: p.name)
def test_no_report_cites_a_number_that_disagrees_with_metrics(document: Path) -> None:
    """The check that stops a regenerated metric from leaving stale prose behind."""
    problems = [
        p
        for p in check_documents([document])[str(document)]
        if p["problem"] != "pointer, no value quoted"
    ]
    assert problems == [], f"{document.name}: {problems}"


def test_the_technical_report_actually_cites_its_numbers() -> None:
    """A report that cites nothing would pass the check above vacuously."""
    report = REPORT_DIR / "TECHNICAL_REPORT.md"
    assert report.exists()
    assert citation_count(report.read_text()) > 50


def test_a_wrong_number_is_caught() -> None:
    metrics = {"a": {"b": 0.537}}
    assert check_text("0.537 <!-- metrics.json: a.b -->", metrics) == []
    assert check_text("0.637 <!-- metrics.json: a.b -->", metrics)[0]["problem"] == (
        "does not match metrics"
    )


def test_a_wrong_sign_is_caught_even_with_a_typographic_minus() -> None:
    """The ToxCast sign bug is the reason this matters."""
    metrics = {"a": {"b": -0.799}}
    assert check_text("−0.799 <!-- metrics.json: a.b -->", metrics) == []
    assert check_text("0.799 <!-- metrics.json: a.b -->", metrics)[0]["problem"] == (
        "does not match metrics"
    )


def test_rounding_and_separators_are_not_discrepancies() -> None:
    metrics = {"n": 21272, "x": 0.53728, "p": 0.992}
    assert check_text("21,272 <!-- metrics.json: n -->", metrics) == []
    assert check_text("0.537 <!-- metrics.json: x -->", metrics) == []
    assert check_text("99.2% <!-- metrics.json: p -->", metrics) == []


def test_an_interval_is_compared_element_wise() -> None:
    metrics = {"ci": [0.4216, 0.6845]}
    assert check_text("[0.4216, 0.6845] <!-- metrics.json: ci -->", metrics) == []
    assert check_text("[0.4216, 0.9] <!-- metrics.json: ci -->", metrics)
    assert check_text("[0.42, 0.68] <!-- metrics.json: ci -->", metrics) == []  # rounded


def test_a_missing_key_is_reported_rather_than_ignored() -> None:
    assert check_text("0.5 <!-- metrics.json: nope.gone -->", {})[0]["problem"] == (
        "key not in metrics"
    )


def test_a_citation_with_no_number_before_it_is_not_silently_accepted() -> None:
    """A number separated from its citation used to be skipped entirely."""
    problems = check_text("some prose <!-- metrics.json: a -->", {"a": 0.5})
    assert problems and problems[0]["problem"] == "does not match metrics"


def test_structured_metrics_are_treated_as_pointers_not_crashes() -> None:
    for value in ({"x": True}, [{"x": 1}], [True, False]):
        verdict, _ = compare("as recorded in ", value)
        assert verdict == "pointer"


def test_lookup_walks_dicts_and_list_indices() -> None:
    metrics = {"a": {"b": [{"c": 7}]}}
    assert lookup(metrics, "a.b.0.c") == 7
    assert lookup(metrics, "a.b.9.c") is None
    assert lookup(metrics, "a.missing") is None


def test_every_cited_key_exists_in_the_live_metrics() -> None:
    """Catches a key renamed by a script without the report following."""
    metrics = json.loads((REPORT_DIR.parent / "outputs" / "metrics.json").read_text())
    missing = []
    for document in REPORTS:
        for problem in check_text(document.read_text(), metrics):
            if problem["problem"] == "key not in metrics":
                missing.append((document.name, problem["key"]))
    assert missing == []


def test_generated_attribution_files_are_up_to_date() -> None:
    """CITATIONS.md and THIRD_PARTY.md are derived; drift means an unrecorded change."""
    from neuromps.report.attribution import main as attribution

    assert attribution(["--check"]) == 0


def test_every_used_source_has_a_recorded_license() -> None:
    """A source without a license must never reach the citations file."""
    from neuromps.report.attribution import manifest_sources

    for name, entry in manifest_sources().items():
        assert entry.get("license"), name


def test_the_excluded_source_is_listed_with_its_reason() -> None:
    """Dropping Tukker 2020 on license grounds is a result, not an omission."""
    from neuromps.report.attribution import CITATIONS, manifest_sources, render_citations

    text = render_citations(manifest_sources())
    assert "Sources deliberately not used" in text
    assert "iras_mea" in text.split("Sources deliberately not used")[1]
    assert CITATIONS.read_text() == text


def test_reports_make_no_clinical_claim() -> None:
    """The project rule, checked rather than trusted.

    The patterns are deliberately narrow: "diagnostics" in the sense of model
    diagnostics is ordinary technical vocabulary and must not trip this.
    """
    import re

    banned = re.compile(
        r"\befficacy\b|\bdiagnosis\b|\bdiagnostic (claim|use|tool|device)\b"
        r"|therapeutic benefit|safe for (human|patient)|treatment of patients",
        re.I,
    )
    # A line may name these words only to rule them out.
    allowed = (
        "no efficacy",
        "no clinical",
        "not an efficacy",
        "claim is made",
        "decision-support",
        "never",
        "diagnostic or",
    )
    for document in REPORTS:
        for line in document.read_text().splitlines():
            if banned.search(line):
                assert any(marker in line.lower() for marker in allowed), (
                    f"{document.name}: {line.strip()[:90]}"
                )


def test_the_claim_check_catches_a_real_claim_and_spares_model_diagnostics() -> None:
    """The check is only worth having if it would fire on the thing it guards against."""
    import re

    banned = re.compile(
        r"\befficacy\b|\bdiagnosis\b|\bdiagnostic (claim|use|tool|device)\b"
        r"|therapeutic benefit|safe for (human|patient)|treatment of patients",
        re.I,
    )
    assert banned.search("this shows efficacy in human cultures")
    assert banned.search("suitable for diagnostic use")
    assert not banned.search("correction 2, diagnostics: rank-test p-values were compared")
    assert not banned.search("posterior diagnostics and calibration checks")


def test_an_empty_collection_is_a_pointer_rather_than_a_crash() -> None:
    """`boundary_hits: []` means "nothing was flagged" - a result worth citing.

    It used to take the whole verification run down: `_trailing_numbers(window, 0)`
    slices `found[-0:]`, which returns every number in the window instead of
    none, and the strict zip that follows then raised ValueError.
    """
    for empty in ([], {}):
        verdict, _ = compare("the optimum is interior rather than clipped", empty)
        assert verdict == "pointer"
    verdict, _ = compare("a search that flagged 3 parameters", [])
    assert verdict == "pointer"

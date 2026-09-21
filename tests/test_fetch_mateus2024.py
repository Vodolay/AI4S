"""The licence handling for a source whose own README contradicts its metadata.

This source is included on the strength of its Zenodo metadata licence plus a
written clarification from the depositor. That is a weaker footing than every
other source in the asset, so the rules around it are tested rather than
trusted: the metadata licence is what gates the fetch, and the README's
contradicting claim has to survive into the provenance record.
"""

import pytest

from neuromps.fetch import mateus2024
from neuromps.fetch.download import SourceBlocked

RESTRICTIVE_README = """WELCOME

This dataset contains all spike data for all microelectrode arrays experiments.

IMPORTANT

THIS DATASET IS PUBLICALLY AVAILABLE UNDER A CREATIVE COMMONS CC BY-NC-ND LICENSE
(ATTRIBUTION-NONCOMMERCIAL-NODERIVS)
"""


def test_the_readme_claim_is_detected_not_ignored() -> None:
    claim = mateus2024.readme_license_claim(RESTRICTIVE_README)
    assert claim is not None
    assert "BY-NC-ND" in claim


def test_a_clean_readme_reports_no_claim() -> None:
    assert mateus2024.readme_license_claim("Spike data. Licensed CC BY 4.0.") is None


@pytest.mark.parametrize(
    "line",
    [
        "Released under CC BY-NC-ND 4.0",
        "This data is NonCommercial only",
        "licence: NoDerivs",
        "cc by-nc-nd",
    ],
)
def test_restrictive_wordings_are_all_caught(line: str) -> None:
    """The README can phrase it several ways; any of them must be recorded."""
    assert mateus2024.readme_license_claim(f"header\n{line}\nfooter") is not None


def test_the_note_carries_both_the_conflict_and_its_resolution() -> None:
    note = mateus2024.license_note(RESTRICTIVE_README)
    assert "CC BY 4.0 only" in note  # the depositor's words
    assert "2026-09-21" in note  # when they said it
    assert "BY-NC-ND" in note  # what the deposit still says
    assert "publicly checkable" in note  # which of the two a reader can verify


def test_the_note_says_so_if_the_readme_is_ever_fixed() -> None:
    note = mateus2024.license_note("Spike data, CC BY 4.0.")
    assert "no longer states a restrictive licence" in note


def test_the_note_does_not_pretend_to_have_read_a_missing_readme() -> None:
    assert "README not read" in mateus2024.license_note(None)


def test_the_expected_licence_is_the_metadata_one() -> None:
    assert mateus2024.EXPECTED_LICENSE == "cc-by-4.0"
    assert mateus2024.META["license"] == "CC-BY-4.0"


def test_fetch_blocks_if_the_metadata_licence_moves(monkeypatch, tmp_path) -> None:
    """Inclusion rests on the metadata licence, so a change there stops the fetch."""
    monkeypatch.setattr(mateus2024.zenodo, "record", lambda _id: {"metadata": {}})
    monkeypatch.setattr(mateus2024.zenodo, "license_id", lambda _rec: "cc-by-nc-nd-4.0")
    with pytest.raises(SourceBlocked, match="license changed upstream"):
        mateus2024.fetch(tmp_path, _EmptyManifest(), full=False)


def test_fetch_blocks_if_a_wanted_file_disappears(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(mateus2024.zenodo, "record", lambda _id: {"metadata": {}})
    monkeypatch.setattr(mateus2024.zenodo, "license_id", lambda _rec: "cc-by-4.0")
    monkeypatch.setattr(mateus2024.zenodo, "files", lambda _rec: {})
    with pytest.raises(SourceBlocked, match="files missing"):
        mateus2024.fetch(tmp_path, _EmptyManifest(), full=False)


def test_the_image_archive_is_deliberately_not_fetched() -> None:
    assert "2_Raw_Images.zip" not in mateus2024.QUICK_FILES + mateus2024.FULL_FILES


class _EmptyManifest:
    """Stand-in manifest with no cached files."""

    def file_index(self) -> dict:
        return {}

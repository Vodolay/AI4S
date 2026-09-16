"""Offline tests for the Zenodo/figshare-backed sources and their license guards."""

from pathlib import Path

import pytest

from neuromps.fetch import fofie2025, screenneuropharm_bic, sharf2022, zenodo
from neuromps.fetch.download import SourceBlocked
from neuromps.fetch.manifest import Manifest
from neuromps.fetch.runner import registry

RECORD = {
    "metadata": {"license": {"id": "cc-zero"}},
    "files": [
        {
            "key": "kilosort2.zip",
            "size": 929_500_000,
            "checksum": "md5:0123456789abcdef0123456789abcdef",
            "links": {"self": "https://zenodo.org/api/records/1/files/kilosort2.zip/content"},
        }
    ],
}


def test_zenodo_files_parses_size_and_md5() -> None:
    f = zenodo.files(RECORD)["kilosort2.zip"]
    assert f.size == 929_500_000
    assert f.md5 == "0123456789abcdef0123456789abcdef"
    assert zenodo.license_id(RECORD) == "cc-zero"


def test_zenodo_md5_absent_is_none() -> None:
    rec = {"files": [{"key": "a", "size": 1, "checksum": "sha256:x", "links": {"self": "u"}}]}
    assert zenodo.files(rec)["a"].md5 is None


def test_sharf_blocks_when_license_changes(tmp_path: Path, monkeypatch) -> None:
    changed = {**RECORD, "metadata": {"license": {"id": "cc-by-nc-4.0"}}}
    monkeypatch.setattr(sharf2022.zenodo, "record", lambda _id: changed)
    with pytest.raises(SourceBlocked, match="license changed upstream"):
        sharf2022.fetch(tmp_path, Manifest.load(tmp_path / "m.json"), full=False)


def test_sharf_blocks_when_expected_file_is_gone(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(sharf2022.zenodo, "record", lambda _id: {**RECORD, "files": []})
    with pytest.raises(SourceBlocked, match="files missing"):
        sharf2022.fetch(tmp_path, Manifest.load(tmp_path / "m.json"), full=False)


def test_fofie_blocks_when_license_changes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(fofie2025, "get_json", lambda _u: {"license": {"name": "CC BY-NC"}})
    with pytest.raises(SourceBlocked, match="license changed upstream"):
        fofie2025.fetch(tmp_path, Manifest.load(tmp_path / "m.json"), full=False)


def test_screenneuropharm_quick_tier_skips_large_archive(tmp_path: Path, monkeypatch) -> None:
    rec = {"metadata": {"license": {"id": "other-open"}}, "files": []}
    monkeypatch.setattr(screenneuropharm_bic.zenodo, "record", lambda _id: rec)
    out = screenneuropharm_bic.fetch(tmp_path, Manifest.load(tmp_path / "m.json"), full=False)
    assert out.status == "skipped"
    assert out.files == []
    assert out.license == "MIT"


def test_all_new_sources_are_registered() -> None:
    assert {"sharf2022", "fofie2025", "screenneuropharm_bic"} <= set(registry())

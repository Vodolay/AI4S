"""Tests for provenance hashing and the raw-data manifest."""

import hashlib
import json
from pathlib import Path

import pytest

from neuromps.fetch.manifest import FileRecord, Manifest, SourceRecord
from neuromps.qc.provenance import sha256_bytes, sha256_file


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    payload = b"spike times\n" * 100_000  # > one read chunk
    f = tmp_path / "x.bin"
    f.write_bytes(payload)
    assert sha256_file(f) == hashlib.sha256(payload).hexdigest()


def test_sha256_empty_file(tmp_path: Path) -> None:
    f = tmp_path / "empty"
    f.write_bytes(b"")
    assert sha256_file(f) == hashlib.sha256(b"").hexdigest()


def test_sha256_bytes_known_vector() -> None:
    assert sha256_bytes(b"abc") == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def _record(name: str = "demo") -> SourceRecord:
    return SourceRecord(
        name=name,
        status="ok",
        landing_url="https://example.org/record/1",
        license="CC-BY-4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        citation="Doe et al. 2022",
        files=[
            FileRecord(
                path="demo/a.csv",
                url="https://example.org/a.csv",
                sha256="0" * 64,
                bytes=3,
                downloaded_at="2026-09-16T00:00:00+00:00",
            )
        ],
    )


def test_manifest_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "MANIFEST.json"
    m = Manifest.load(path)
    m.upsert(_record())
    m.save()
    data = json.loads(path.read_text())
    assert data["sources"]["demo"]["license"] == "CC-BY-4.0"
    reloaded = Manifest.load(path)
    assert reloaded.sources["demo"].files[0].sha256 == "0" * 64


def test_manifest_upsert_replaces_and_keeps_others(tmp_path: Path) -> None:
    m = Manifest.load(tmp_path / "MANIFEST.json")
    m.upsert(_record("a"))
    m.upsert(_record("b"))
    blocked = _record("a")
    blocked.status = "blocked"
    blocked.files = []
    m.upsert(blocked)
    assert set(m.sources) == {"a", "b"}
    assert m.sources["a"].status == "blocked"


def test_ok_source_requires_license() -> None:
    rec = _record()
    rec.license = ""
    with pytest.raises(ValueError, match="license"):
        rec.validate()


def test_ok_source_requires_files() -> None:
    rec = _record()
    rec.files = []
    with pytest.raises(ValueError, match="files"):
        rec.validate()


def test_file_record_rejects_bad_hash() -> None:
    with pytest.raises(ValueError, match="sha256"):
        FileRecord(path="p", url="u", sha256="xyz", bytes=1, downloaded_at="t").validate()


def test_excluded_sources_are_recorded_without_files(tmp_path: Path) -> None:
    from neuromps.fetch import runner
    from neuromps.fetch.excluded import EXCLUDED

    runner_registry = runner.registry
    runner.registry = lambda: {}  # no network: only exclusions are written
    try:
        m = runner.run_fetchers(raw_root=tmp_path)
    finally:
        runner.registry = runner_registry
    for name in EXCLUDED:
        assert m.sources[name].status == "skipped"
        assert m.sources[name].files == []
        assert name not in runner.registry()
    assert json.loads((tmp_path / "MANIFEST.json").read_text())["sources"]["iras_mea"]["note"]

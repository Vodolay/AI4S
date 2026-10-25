"""The male CNS neuron table is checked against the bucket's CRC32C, and is optional."""

from __future__ import annotations

import base64

import pytest

from neuromps.fetch import crc32c, malecns_sizes, runner
from neuromps.fetch.download import SourceBlocked
from neuromps.fetch.manifest import FileRecord, Manifest


def _fake_download(body: bytes):
    def fake(url, raw_root, rel, *, cached=None, max_bytes=None, **kwargs):
        path = raw_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        return FileRecord(rel, url, "a" * 64, len(body), "t")

    return fake


def _published(value: int) -> dict:
    return {"size": "9", "generation": "1", "crc32c": base64.b64encode(value.to_bytes(4, "big"))}


def test_a_matching_crc32c_records_the_source(monkeypatch, tmp_path) -> None:
    body = b"123456789"
    monkeypatch.setattr(malecns_sizes, "published", lambda: _published(crc32c.crc32c(body)))
    monkeypatch.setattr(malecns_sizes, "download_file", _fake_download(body))
    record = malecns_sizes.fetch(tmp_path, Manifest(tmp_path / "MANIFEST.json"), full=False)
    assert record.status == "ok" and record.files[0].path == malecns_sizes.REL
    assert "CRC32C" in record.note and record.license.startswith("CC-BY-4.0")


def test_a_mismatch_deletes_the_file_and_blocks(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(malecns_sizes, "published", lambda: _published(0))
    monkeypatch.setattr(malecns_sizes, "download_file", _fake_download(b"123456789"))
    with pytest.raises(SourceBlocked, match="CRC32C mismatch"):
        malecns_sizes.fetch(tmp_path, Manifest(tmp_path / "MANIFEST.json"), full=False)
    assert not (tmp_path / malecns_sizes.REL).exists()


def test_no_published_crc32c_blocks(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(malecns_sizes, "published", lambda: {"size": "9"})
    with pytest.raises(SourceBlocked, match="no CRC32C"):
        malecns_sizes.fetch(tmp_path, Manifest(tmp_path / "MANIFEST.json"), full=False)


def test_the_sizes_are_optional() -> None:
    assert malecns_sizes.SOURCE in runner.OPTIONAL

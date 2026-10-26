"""Provenance for the invitrodb MySQL dump: size check, hash, manifest entry, resume."""

import hashlib

import pytest

from neuromps.fetch import toxcast_mysql as tm
from neuromps.fetch.manifest import FileRecord, Manifest, SourceRecord
from neuromps.fetch.toxcast_mea import META, SOURCE


def _dump(tmp_path, data=b"-- fake dump\n"):
    p = tmp_path / tm.REL_PATH
    p.parent.mkdir(parents=True)
    p.write_bytes(data)
    return p, data


def test_record_adds_file_to_existing_source(tmp_path):
    _, data = _dump(tmp_path)
    m = Manifest(path=tmp_path / "MANIFEST.json")
    old = FileRecord("toxcast_mea/a.pdf", "u", "0" * 64, 1, "2026-01-01T00:00:00+00:00")
    m.upsert(SourceRecord(name=SOURCE, status="ok", files=[old], **META))
    rec = tm.record(m, tmp_path, "2026-10-26T08:00:00+00:00", expected_bytes=len(data))
    assert rec.sha256 == hashlib.sha256(data).hexdigest()
    assert rec.url == tm.URL and rec.bytes == len(data)
    paths = [f.path for f in m.sources[SOURCE].files]
    assert paths == ["toxcast_mea/a.pdf", tm.REL_PATH]
    # recording again replaces, does not duplicate
    tm.record(m, tmp_path, "2026-10-26T09:00:00+00:00", expected_bytes=len(data))
    assert [f.path for f in m.sources[SOURCE].files] == paths
    assert m.sources[SOURCE].license == "CC0-1.0"


def test_record_refuses_wrong_size(tmp_path):
    _, data = _dump(tmp_path)
    m = Manifest(path=tmp_path / "MANIFEST.json")
    with pytest.raises(ValueError, match="expected"):
        tm.record(m, tmp_path, "t", expected_bytes=len(data) + 1)


class _Resp:
    def __init__(self, status, body):
        self.status_code, self.body = status, body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)

    def iter_content(self, n):
        yield self.body


class _Session:
    def __init__(self, full, status=206):
        self.full, self.status, self.ranges = full, status, []

    def get(self, url, headers, stream, timeout):
        start = int(headers["Range"].split("=")[1].rstrip("-"))
        self.ranges.append(start)
        return _Resp(self.status, self.full[start:] if self.status == 206 else self.full)


def test_download_resumes_from_partial(tmp_path):
    full = bytes(range(200))
    dest = tmp_path / "x.gz"
    dest.write_bytes(full[:70])
    s = _Session(full)
    assert tm.download(dest, expected_bytes=len(full), session=s) == len(full)
    assert s.ranges == [70] and dest.read_bytes() == full


def test_download_rejects_ignored_range(tmp_path):
    full = bytes(range(50))
    dest = tmp_path / "x.gz"
    dest.write_bytes(full[:10])
    with pytest.raises(RuntimeError, match="ignored the range"):
        tm.download(dest, expected_bytes=len(full), session=_Session(full, status=200))

"""Tests for the streaming downloader (no network: uses a local HTTP server)."""

import hashlib
import http.server
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from neuromps.fetch.download import DownloadTooLarge, download_file


@pytest.fixture()
def http_root(tmp_path: Path) -> Iterator[tuple[str, Path]]:
    root = tmp_path / "srv"
    root.mkdir()

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):  # type: ignore[no-untyped-def]
            super().__init__(*a, directory=str(root), **kw)

        def log_message(self, *a) -> None:  # type: ignore[no-untyped-def]
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", root
    server.shutdown()


def test_download_writes_file_and_hash(http_root: tuple[str, Path], tmp_path: Path) -> None:
    base, root = http_root
    payload = b"a,b\n1,2\n" * 5000
    (root / "t.csv").write_bytes(payload)
    rec = download_file(f"{base}/t.csv", tmp_path / "out", "src/t.csv")
    assert (tmp_path / "out" / "src" / "t.csv").read_bytes() == payload
    assert rec.sha256 == hashlib.sha256(payload).hexdigest()
    assert rec.bytes == len(payload)
    assert rec.path == "src/t.csv"


def test_download_skips_when_hash_matches(http_root: tuple[str, Path], tmp_path: Path) -> None:
    base, root = http_root
    (root / "t.csv").write_bytes(b"x")
    out = tmp_path / "out"
    first = download_file(f"{base}/t.csv", out, "t.csv")
    (root / "t.csv").write_bytes(b"changed upstream")
    second = download_file(f"{base}/t.csv", out, "t.csv", cached=first)
    assert second.downloaded_at == first.downloaded_at  # cached, not refetched
    assert (out / "t.csv").read_bytes() == b"x"


def test_download_rejects_md5_mismatch(http_root: tuple[str, Path], tmp_path: Path) -> None:
    base, root = http_root
    (root / "t.csv").write_bytes(b"x")
    with pytest.raises(ValueError, match="md5"):
        download_file(f"{base}/t.csv", tmp_path, "t.csv", expected_md5="0" * 32)
    assert not (tmp_path / "t.csv").exists()


def test_download_size_cap(http_root: tuple[str, Path], tmp_path: Path) -> None:
    base, root = http_root
    (root / "big").write_bytes(b"0" * 10_000)
    with pytest.raises(DownloadTooLarge):
        download_file(f"{base}/big", tmp_path, "big", max_bytes=1000)
    assert not (tmp_path / "big").exists()


def test_fetch_many_preserves_request_order_when_parallel(tmp_path, monkeypatch) -> None:
    """The manifest must not depend on which download finished first."""
    import random
    import time

    from neuromps.fetch import common

    def slow_download(url, raw_root, rel, cached=None, max_bytes=None, expected_md5=None):
        time.sleep(random.uniform(0, 0.05))
        return common.FileRecord(path=rel, url=url, sha256="x" * 64, bytes=1, downloaded_at="t")

    monkeypatch.setattr(common, "download_file", slow_download)
    items = [(f"http://example/{i}", f"src/file_{i:02d}.txt") for i in range(12)]
    records = common.fetch_many(items, tmp_path, cached={}, workers=6)
    assert [r.path for r in records] == [rel for _, rel in items]


def test_fetch_many_serial_and_parallel_agree(tmp_path, monkeypatch) -> None:
    from neuromps.fetch import common

    def fake(url, raw_root, rel, cached=None, max_bytes=None, expected_md5=None):
        return common.FileRecord(path=rel, url=url, sha256="y" * 64, bytes=2, downloaded_at="t")

    monkeypatch.setattr(common, "download_file", fake)
    items = [(f"http://example/{i}", f"src/f{i}.txt") for i in range(5)]
    serial = common.fetch_many(items, tmp_path, cached={}, workers=1)
    parallel = common.fetch_many(items, tmp_path, cached={}, workers=4)
    assert [r.path for r in serial] == [r.path for r in parallel]

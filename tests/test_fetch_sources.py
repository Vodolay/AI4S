"""Offline tests for source-specific fetch helpers."""

import gzip
import http.server
import io
import threading
import zipfile
import zlib
from collections.abc import Iterator
from pathlib import Path

import pytest

from neuromps.fetch import kapucu2022
from neuromps.fetch.download import SourceBlocked
from neuromps.fetch.europepmc import require_open_license
from neuromps.fetch.remote_zip import extract_member

CC_BY_XML = """<article><front><article-meta><permissions>
<license license-type="open-access" xlink:href="http://creativecommons.org/licenses/by/4.0/"
 xmlns:xlink="http://www.w3.org/1999/xlink"><license-p>CC BY</license-p></license>
</permissions></article-meta></front></article>"""
NO_LICENSE_XML = "<article><front><article-meta></article-meta></front></article>"


def test_require_open_license_accepts_cc_by() -> None:
    assert require_open_license(CC_BY_XML) == "http://creativecommons.org/licenses/by/4.0/"


def test_require_open_license_blocks_missing() -> None:
    with pytest.raises(SourceBlocked):
        require_open_license(NO_LICENSE_XML)


def test_kapucu_file_list_tiers_exclude_raw() -> None:
    quick = kapucu2022.file_list(full=False)
    full = kapucu2022.file_list(full=True)
    assert set(quick) < set(full)
    assert not any(p.endswith(".h5") for p in full)
    assert "Data/LICENSE.txt" in quick
    assert any("Pharmacology" in p and p.endswith("_spikes.csv") for p in quick)


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    payload = b""

    def do_GET(self) -> None:  # noqa: N802
        data, rng = self.payload, self.headers.get("Range")
        if rng:
            start_s, end_s = rng.removeprefix("bytes=").split("-")
            if not start_s:  # mimic EPA Clowder, which fails on suffix ranges
                self.send_error(500)
                return
            if "gzip" in self.headers.get("Accept-Encoding", ""):
                # mimic EPA Clowder: claims 206 but streams the whole file gzip-encoded
                body = gzip.compress(data)
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start_s}-{end_s}/{len(data)}")
                self.send_header("Content-Encoding", "gzip")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            start = int(start_s)
            end = int(end_s) if end_s else len(data) - 1
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
            chunk = data[start : end + 1]
        else:
            self.send_response(200)
            chunk = data
        self.send_header("Content-Length", str(len(chunk)))
        self.end_headers()
        self.wfile.write(chunk)

    def log_message(self, *a) -> None:  # type: ignore[no-untyped-def]
        pass


@pytest.fixture()
def zip_server() -> Iterator[tuple[str, bytes]]:
    member = b"aeid,chid,hitc\n" + b"2454,1,0.9\n" * 50_000
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("other.txt", b"x" * 1000)
        z.writestr("mc5.csv", member)
    _RangeHandler.payload = buf.getvalue()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/z.zip", member
    server.shutdown()


def test_extract_member_over_http_ranges(zip_server: tuple[str, bytes], tmp_path: Path) -> None:
    url, member = zip_server
    rec = extract_member(url, "mc5.csv", tmp_path, "tox/mc5.csv")
    assert (tmp_path / "tox" / "mc5.csv").read_bytes() == member
    assert rec.bytes == len(member)
    assert zlib.crc32(member) == zlib.crc32((tmp_path / "tox" / "mc5.csv").read_bytes())
    assert "#member=mc5.csv" in rec.url


def test_extract_member_missing_name(zip_server: tuple[str, bytes], tmp_path: Path) -> None:
    url, _ = zip_server
    with pytest.raises(KeyError):
        extract_member(url, "nope.csv", tmp_path, "nope.csv")

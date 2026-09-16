"""Extract one member of a remote ZIP via HTTP range requests (no full download)."""

import hashlib
import io
import zipfile
from pathlib import Path

import requests

from neuromps.fetch.download import USER_AGENT
from neuromps.fetch.manifest import FileRecord, utc_now

_CHUNK = 8 << 20


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP ranges.

    Sequential reads reuse one open streaming response, so extracting a large
    member costs one request plus a few for the central directory.
    """

    def __init__(self, url: str, timeout: float = 300) -> None:
        self.url, self.timeout, self.pos = url, timeout, 0
        self._resp: requests.Response | None = None
        self._iter = None
        self._stream_pos = -1
        self._buf = b""
        # Explicit range: EPA Clowder answers suffix ranges ("bytes=-1") with HTTP 500.
        with self._get_range(0, 0) as probe:
            total = probe.headers["Content-Range"].rpartition("/")[2]
        if not total.isdigit():
            raise OSError(f"unknown total size in Content-Range: {url}")
        self.size = int(total)

    def _get_range(self, start: int, end: int | None = None) -> requests.Response:
        """Open a streaming range response and verify the server honoured the range.

        ``Accept-Encoding: identity`` is required: when gzip is offered, EPA Clowder
        replies 206 with a correct-looking Content-Range but streams the whole
        file gzip-encoded from byte 0.
        """
        rng = f"bytes={start}-{'' if end is None else end}"
        r = requests.get(
            self.url,
            headers={"User-Agent": USER_AGENT, "Range": rng, "Accept-Encoding": "identity"},
            stream=True,
            timeout=self.timeout,
        )
        r.raise_for_status()
        content_range = r.headers.get("Content-Range", "")
        ok = (
            r.status_code == 206
            and content_range.startswith(f"bytes {start}-")
            and r.headers.get("Content-Encoding", "identity") == "identity"
        )
        if not ok:
            r.close()
            raise OSError(
                f"range {rng} not honoured: status={r.status_code} "
                f"Content-Range={content_range!r} "
                f"Content-Encoding={r.headers.get('Content-Encoding')}"
            )
        return r

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def _open_at(self, pos: int) -> None:
        if self._resp is not None:
            self._resp.close()
        self._resp = self._get_range(pos)
        self._iter = self._resp.iter_content(_CHUNK)
        self._stream_pos, self._buf = pos, b""

    def readinto(self, b: bytearray) -> int:  # type: ignore[override]
        if self.pos >= self.size:
            return 0
        if self._stream_pos != self.pos:
            self._open_at(self.pos)
        while not self._buf:
            self._buf = next(self._iter, b"")  # type: ignore[arg-type]
            if not self._buf:
                return 0
        n = min(len(b), len(self._buf))
        b[:n] = self._buf[:n]
        self._buf = self._buf[n:]
        self.pos += n
        self._stream_pos = self.pos
        return n

    def close(self) -> None:
        if self._resp is not None:
            self._resp.close()
        super().close()


def extract_member(url: str, member: str, raw_root: Path, rel_path: str) -> FileRecord:
    """Stream-decompress ``member`` of the remote zip at ``url`` to ``raw_root/rel_path``.

    zipfile verifies the member CRC32 on read; a mismatch raises and the
    partial file is removed. Raises KeyError if the member does not exist.
    """
    dest = Path(raw_root) / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    sha, n = hashlib.sha256(), 0
    raw = HttpRangeFile(url)
    try:
        with (
            zipfile.ZipFile(io.BufferedReader(raw, _CHUNK)) as zf,
            zf.open(member) as src,
            part.open("wb") as out,
        ):
            while chunk := src.read(_CHUNK):
                sha.update(chunk)
                out.write(chunk)
                n += len(chunk)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    finally:
        raw.close()
    part.replace(dest)
    return FileRecord(
        path=rel_path,
        url=f"{url}#member={member}",
        sha256=sha.hexdigest(),
        bytes=n,
        downloaded_at=utc_now(),
    )

"""Streaming HTTP download with hashing, size cap and cache reuse."""

import hashlib
import time
from pathlib import Path

import requests

from neuromps.fetch.manifest import FileRecord, utc_now
from neuromps.qc.provenance import sha256_file

USER_AGENT = "neuromps-fetch/0.0.1 (research data pipeline)"
_CHUNK = 1 << 20


class DownloadTooLarge(RuntimeError):
    """Raised when a file exceeds the configured size cap."""


class SourceBlocked(RuntimeError):
    """Raised by a fetcher when a source cannot be retrieved without gating."""


def get_json(url: str, timeout: float = 60, retries: int = 3) -> dict:
    """GET a JSON document with simple retry/backoff."""
    for attempt in range(retries):
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def download_file(
    url: str,
    raw_root: Path,
    rel_path: str,
    *,
    cached: FileRecord | None = None,
    expected_md5: str | None = None,
    max_bytes: int | None = None,
    timeout: float = 120,
) -> FileRecord:
    """Download ``url`` to ``raw_root/rel_path`` and return its provenance record.

    If ``cached`` describes the same path and the on-disk SHA256 still matches,
    the file is not re-downloaded. Checksum or size-cap failures delete the
    partial file and raise.
    """
    dest = Path(raw_root) / rel_path
    if cached and cached.path == rel_path and dest.exists() and sha256_file(dest) == cached.sha256:
        return cached
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    sha, md5, n = hashlib.sha256(), hashlib.md5(), 0
    try:
        with requests.get(
            url, stream=True, timeout=timeout, headers={"User-Agent": USER_AGENT}
        ) as r:
            r.raise_for_status()
            with part.open("wb") as fh:
                for chunk in r.iter_content(_CHUNK):
                    n += len(chunk)
                    if max_bytes is not None and n > max_bytes:
                        raise DownloadTooLarge(f"{url} exceeds {max_bytes} bytes")
                    sha.update(chunk)
                    md5.update(chunk)
                    fh.write(chunk)
        if expected_md5 and md5.hexdigest() != expected_md5:
            raise ValueError(f"md5 mismatch for {url}: {md5.hexdigest()} != {expected_md5}")
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    part.replace(dest)
    return FileRecord(
        path=rel_path, url=url, sha256=sha.hexdigest(), bytes=n, downloaded_at=utc_now()
    )

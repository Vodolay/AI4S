"""Streaming HTTP download with hashing, size cap and cache reuse."""

import hashlib
import logging
import time
from pathlib import Path

import requests

from neuromps.fetch.manifest import FileRecord, utc_now
from neuromps.qc.provenance import sha256_file

USER_AGENT = "neuromps-fetch/0.0.1 (research data pipeline)"
_CHUNK = 1 << 20
log = logging.getLogger("neuromps.fetch")


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


# Statuses a busy host uses to say "not now". 403 is in the set deliberately:
# gin.g-node.org answers a burst of requests from a CI address with 403 rather
# than 429, and that stopped a pipeline run minutes after the same files
# downloaded cleanly from a laptop. A fetcher marks its whole source blocked on
# the first failure, so these retries cost one file's worth of waiting, not the
# whole list's.
RETRY_STATUSES = frozenset({403, 408, 429, 500, 502, 503, 504})
RETRY_ATTEMPTS = 4
RETRY_BASE_S = 2.0


def get_with_retry(
    url: str,
    timeout: float,
    attempts: int = RETRY_ATTEMPTS,
    sleep=time.sleep,
) -> requests.Response:
    """A streaming GET, backing off while the host is refusing to serve us."""
    for attempt in range(attempts):
        response = requests.get(
            url, stream=True, timeout=timeout, headers={"User-Agent": USER_AGENT}
        )
        last = attempt == attempts - 1
        if response.status_code not in RETRY_STATUSES or last:
            return response
        response.close()
        delay = RETRY_BASE_S * 3**attempt
        log.warning(
            "%s returned %d; waiting %.0f s before attempt %d of %d",
            url,
            response.status_code,
            delay,
            attempt + 2,
            attempts,
        )
        sleep(delay)
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
        with get_with_retry(url, timeout) as r:
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

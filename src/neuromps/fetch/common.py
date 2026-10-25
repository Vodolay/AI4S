"""Helpers shared by source fetchers."""

import logging
from pathlib import Path

from neuromps.fetch.download import download_file
from neuromps.fetch.manifest import FileRecord, Manifest
from neuromps.qc.provenance import sha256_file

log = logging.getLogger("neuromps.fetch")


class _Skipped(Exception):
    """A queued download not started because another one had already failed."""


def fetch_many(
    items: list[tuple[str, str]],
    raw_root: Path,
    cached: dict[str, FileRecord],
    max_bytes: int | None = 2 << 30,
    md5s: dict[str, str | None] | None = None,
    workers: int = 4,
) -> list[FileRecord]:
    """Download ``(url, rel_path)`` pairs, reusing files whose hash matches the manifest.

    ``md5s`` maps rel_path to an upstream-published MD5; when given, a mismatch
    deletes the partial file and raises.

    Downloads run on a small thread pool because they are latency-bound, not
    bandwidth-bound: one source publishes 169 separate files, and fetching them
    one at a time dominated a cold run of the pipeline. Four, not more: these
    are small academic servers, and one of them answers a burst from a single
    address by refusing it. Results keep the order
    they were requested in, so the manifest does not depend on which download
    finished first. Set ``workers=1`` for serial behaviour.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    # Set by the first failure. Cancelling queued futures from the main thread is
    # a race: when every download fails at once, the workers can drain the whole
    # queue before the main thread wakes to cancel it, so each worker checks this
    # before it starts a file.
    stop = threading.Event()

    def one(item: tuple[int, tuple[str, str]]) -> tuple[int, FileRecord]:
        index, (url, rel) = item
        if stop.is_set():
            raise _Skipped(rel)
        try:
            record = download_file(
                url,
                raw_root,
                rel,
                cached=cached.get(rel),
                max_bytes=max_bytes,
                expected_md5=(md5s or {}).get(rel),
            )
        except BaseException:
            stop.set()
            raise
        status = "cached" if record is cached.get(rel) else f"{record.bytes / 1e6:.1f} MB"
        log.info("  [%d/%d] %s (%s)", index + 1, len(items), rel, status)
        return index, record

    if workers <= 1 or len(items) <= 1:
        return [one((i, item))[1] for i, item in enumerate(items)]
    pool = ThreadPoolExecutor(max_workers=workers)
    futures = [pool.submit(one, item) for item in enumerate(items)]
    try:
        done = [future.result() for future in futures]
    except BaseException as first:
        # Cancel what has not started. Leaving the pool's context used to wait
        # for every queued file first, so a host that hung on each connection
        # took hours to report the failure the first file had already shown.
        pool.shutdown(wait=True, cancel_futures=True)
        if isinstance(first, _Skipped):
            # An earlier file was skipped after a later one failed: report the failure.
            for future in futures:
                if future.done() and not future.cancelled():
                    error = future.exception()
                    if error is not None and not isinstance(error, _Skipped):
                        raise error from None
        raise
    pool.shutdown()
    return [record for _, record in sorted(done, key=lambda pair: pair[0])]


def retain_existing(
    source: str, manifest: Manifest, raw_root: Path, fetched: list[FileRecord]
) -> list[FileRecord]:
    """Keep earlier records (e.g. from a --full run) whose files are still intact on disk."""
    have = {r.path for r in fetched}
    kept = list(fetched)
    prev = manifest.sources.get(source)
    for rec in prev.files if prev else []:
        p = raw_root / rec.path
        if rec.path not in have and p.exists() and sha256_file(p) == rec.sha256:
            kept.append(rec)
    return sorted(kept, key=lambda r: r.path)

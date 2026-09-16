"""Helpers shared by source fetchers."""

import logging
from pathlib import Path

from neuromps.fetch.download import download_file
from neuromps.fetch.manifest import FileRecord, Manifest
from neuromps.qc.provenance import sha256_file

log = logging.getLogger("neuromps.fetch")


def fetch_many(
    items: list[tuple[str, str]],
    raw_root: Path,
    cached: dict[str, FileRecord],
    max_bytes: int | None = 2 << 30,
    md5s: dict[str, str | None] | None = None,
) -> list[FileRecord]:
    """Download ``(url, rel_path)`` pairs, reusing files whose hash matches the manifest.

    ``md5s`` maps rel_path to an upstream-published MD5; when given, a mismatch
    deletes the partial file and raises.
    """
    records = []
    for i, (url, rel) in enumerate(items, 1):
        rec = download_file(
            url,
            raw_root,
            rel,
            cached=cached.get(rel),
            max_bytes=max_bytes,
            expected_md5=(md5s or {}).get(rel),
        )
        status = "cached" if rec is cached.get(rel) else f"{rec.bytes / 1e6:.1f} MB"
        log.info("  [%d/%d] %s (%s)", i, len(items), rel, status)
        records.append(rec)
    return records


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

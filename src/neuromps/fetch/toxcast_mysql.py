"""The full ToxCast invitrodb v4.3 MySQL dump (17.65 GB gzip), for well-level MEA rows.

Run: ``python -m neuromps.fetch.toxcast_mysql [--download] [--record --downloaded-at ISO]``

Not part of the default fetch tiers: the summary files fetched by
:mod:`neuromps.fetch.toxcast_mea` stop at concentration-response fits, and only
this dump holds the per-well level-0 table (``mc0``) with plate IDs and the
vehicle (``wllt = 'n'``) wells. The file is recorded under the existing
``toxcast_mea`` source (same release, same CC0 licence). It is read as text by
:mod:`neuromps.etl.toxcast_wells`; it is never loaded into a database or run.
"""

import argparse
import logging
from pathlib import Path

import requests

from neuromps.fetch.download import USER_AGENT
from neuromps.fetch.manifest import FileRecord, Manifest, SourceRecord, utc_now
from neuromps.fetch.toxcast_mea import CLOWDER, META, SOURCE
from neuromps.qc.provenance import sha256_file

FILE_ID = "68c3365ce4b02565fc7cd3f3"
DATASET = "687e3940e4b02565bc3e291e"  # Clowder dataset "MySQL_Data", space "invitrodb v4.3"
URL = f"{CLOWDER}/files/{FILE_ID}/blob"
REL_PATH = f"{SOURCE}/mysql/invitrodb_v4_3.sql.gz"
EXPECTED_BYTES = 17_651_807_605
_CHUNK = 1 << 22
log = logging.getLogger("neuromps.fetch")


def download(dest: Path, expected_bytes: int = EXPECTED_BYTES, session=None) -> int:
    """Download ``URL`` to ``dest``, resuming from a partial file; return its size.

    Each attempt asks for the remaining range; a server answering 200 instead
    of 206 would resend the whole file, so that is an error, not a restart.
    """
    session = session or requests.Session()
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0
    if have < expected_bytes:
        headers = {"User-Agent": USER_AGENT, "Range": f"bytes={have}-"}
        with session.get(URL, headers=headers, stream=True, timeout=120) as r:
            if r.status_code != 206 and have:
                raise RuntimeError(f"server ignored the range request (HTTP {r.status_code})")
            r.raise_for_status()
            with dest.open("ab") as fh:
                for chunk in r.iter_content(_CHUNK):
                    fh.write(chunk)
    size = dest.stat().st_size
    if size != expected_bytes:
        raise RuntimeError(f"{dest}: {size} bytes, expected {expected_bytes}")
    return size


def record(
    manifest: Manifest,
    raw_root: Path,
    downloaded_at: str,
    expected_bytes: int = EXPECTED_BYTES,
    sha256: str | None = None,
) -> FileRecord:
    """Add the dump to the ``toxcast_mea`` source in ``manifest`` (not saved).

    The size must match the server's listing. ``sha256`` may be passed when it
    was just computed, to avoid hashing 17.6 GB twice.
    """
    path = Path(raw_root) / REL_PATH
    size = path.stat().st_size
    if size != expected_bytes:
        raise ValueError(f"{path}: {size} bytes, expected {expected_bytes}")
    rec = FileRecord(REL_PATH, URL, sha256 or sha256_file(path), size, downloaded_at)
    src = manifest.sources.get(SOURCE) or SourceRecord(name=SOURCE, status="ok", **META)
    files = [f for f in src.files if f.path != REL_PATH] + [rec]
    src.files = sorted(files, key=lambda f: f.path)
    src.checked_at = utc_now()
    manifest.upsert(src)
    return rec


def main(argv: list[str] | None = None) -> None:
    """CLI: download and/or record the dump in ``data/raw/MANIFEST.json``."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--downloaded-at", default=None, help="ISO time the transfer finished")
    ap.add_argument("--sha256", default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if a.download:
        log.info("%s: %d bytes", REL_PATH, download(a.raw_root / REL_PATH))
    if a.record:
        manifest = Manifest.load(a.manifest)
        rec = record(manifest, a.raw_root, a.downloaded_at or utc_now(), sha256=a.sha256)
        manifest.save()
        log.info("recorded %s sha256=%s bytes=%d", rec.path, rec.sha256, rec.bytes)


if __name__ == "__main__":
    main()

"""ScreenNeuroPharm bicuculline dataset: rat cortical/hippocampal MEA spike trains.

A rat-side GABA-A antagonist concentration series to pair with Kapucu's single
gabazine condition. Licensed MIT (verified in the source repository), which is
permissive and allows commercial reuse; Zenodo only records it as "other-open".
"""

import logging
from pathlib import Path

from neuromps.fetch import zenodo
from neuromps.fetch.common import fetch_many, retain_existing
from neuromps.fetch.download import SourceBlocked
from neuromps.fetch.manifest import FileRecord, Manifest, SourceRecord, utc_now
from neuromps.qc.provenance import sha256_file

log = logging.getLogger("neuromps.fetch")
SOURCE = "screenneuropharm_bic"
RECORD_ID = 7665283
EXPECTED_LICENSE = "other-open"
META = {
    "landing_url": "https://doi.org/10.5281/zenodo.7665283",
    "license": "MIT",
    "license_url": "https://github.com/ScreenNeuroPharm/Bicuculline/blob/main/LICENSE",
    "citation": (
        "ScreenNeuroPharm. Bicuculline (v1). Zenodo. doi:10.5281/zenodo.7665283. "
        "Source repository: github.com/ScreenNeuroPharm/Bicuculline (MIT License, 2023)."
    ),
}


def _adopt_on_disk(raw_root: Path, available: dict) -> list[FileRecord]:
    """Rebuild file records for archives already on disk but absent from the manifest.

    A previous quick run could have dropped them; hashing what is there restores
    provenance without re-downloading 582 MB.
    """
    directory = Path(raw_root) / SOURCE
    urls = {Path(name).name: f.url for name, f in available.items()}
    records = []
    for path in sorted(directory.glob("*.zip")) if directory.exists() else []:
        records.append(
            FileRecord(
                path=f"{SOURCE}/{path.name}",
                url=urls.get(path.name, f"https://doi.org/10.5281/zenodo.{RECORD_ID}"),
                sha256=sha256_file(path),
                bytes=path.stat().st_size,
                downloaded_at=utc_now(),
            )
        )
    if records:
        log.info("[%s] adopted %d file(s) already on disk", SOURCE, len(records))
    return records


def fetch(raw_root: Path, manifest: Manifest, full: bool) -> SourceRecord:
    """Download the spike-train archive (~582 MB) when ``full``; metadata only otherwise."""
    rec = zenodo.record(RECORD_ID)
    got = zenodo.license_id(rec)
    if got != EXPECTED_LICENSE:
        raise SourceBlocked(f"license changed upstream: expected {EXPECTED_LICENSE!r}, got {got!r}")
    available = zenodo.files(rec)
    if not full:
        # Keep whatever a previous --full run left on disk: marking the source
        # "skipped" would drop its files from the manifest, and records already
        # built from them would stop being traceable (conformance level 2).
        kept = retain_existing(SOURCE, manifest, raw_root, []) or _adopt_on_disk(
            raw_root, available
        )
        if kept:
            return SourceRecord(
                name=SOURCE,
                status="ok",
                files=kept,
                note="tier=quick: kept files from an earlier --full fetch; nothing re-downloaded",
                **META,
            )
        return SourceRecord(
            name=SOURCE,
            status="skipped",
            note=(
                "tier=quick: archive is ~582 MB, fetched only with --full. "
                f"Zenodo record {RECORD_ID} lists: {sorted(available)}"
            ),
            **META,
        )
    cached = manifest.file_index()
    items = [(f.url, f"{SOURCE}/{Path(name).name}") for name, f in available.items()]
    md5s = {f"{SOURCE}/{Path(name).name}": f.md5 for name, f in available.items()}
    files = fetch_many(items, raw_root, cached, max_bytes=2 << 30, md5s=md5s)
    return SourceRecord(
        name=SOURCE,
        status="ok",
        files=retain_existing(SOURCE, manifest, raw_root, files),
        note=(
            "tier=full; rat cortical (Cx) and hippocampal (Hp) cultures, 60-electrode MCS MEA, "
            "spike trains; MIT license verified in the source repository, not on Zenodo"
        ),
        **META,
    )

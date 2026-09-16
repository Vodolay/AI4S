"""Sharf et al. 2022: human iPSC brain organoids on a MaxWell HD-MEA, with a diazepam series.

Adds the human dose-response arm the asset otherwise lacks: control, 3, 10, 30
and 50 uM diazepam across 4 organoids. Only the spike-sorted Kilosort2 output is
fetched (~930 MB); the 36 raw .h5 recordings (~60 GB) are left alone.
"""

from pathlib import Path

from neuromps.fetch import zenodo
from neuromps.fetch.common import fetch_many, retain_existing
from neuromps.fetch.download import SourceBlocked
from neuromps.fetch.manifest import Manifest, SourceRecord

SOURCE = "sharf2022"
RECORD_ID = 6578989
EXPECTED_LICENSE = "cc-zero"
QUICK_FILES = ("README_organoid_data.txt",)
FULL_FILES = ("kilosort2.zip",)
META = {
    "landing_url": "https://doi.org/10.25349/D9031Z",
    "license": "CC0-1.0",
    "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
    "citation": (
        "Sharf T, van der Molen T, Glasauer SMK, et al. Functional neuronal circuitry and "
        "oscillatory dynamics in human brain organoids. Dataset: Extracellular Recordings from "
        "Human Brain Organoids Using High-density CMOS Arrays. doi:10.25349/D9031Z"
    ),
}


def fetch(raw_root: Path, manifest: Manifest, full: bool) -> SourceRecord:
    """Download the README, plus the Kilosort2 spike output when ``full``."""
    rec = zenodo.record(RECORD_ID)
    got = zenodo.license_id(rec)
    if got != EXPECTED_LICENSE:
        raise SourceBlocked(f"license changed upstream: expected {EXPECTED_LICENSE!r}, got {got!r}")
    available = zenodo.files(rec)
    wanted = QUICK_FILES + (FULL_FILES if full else ())
    missing = [n for n in wanted if n not in available]
    if missing:
        raise SourceBlocked(f"files missing from record {RECORD_ID}: {missing}")
    cached = manifest.file_index()
    items = [(available[n].url, f"{SOURCE}/{n}") for n in wanted]
    md5s = {f"{SOURCE}/{n}": available[n].md5 for n in wanted}
    files = fetch_many(items, raw_root, cached, max_bytes=2 << 30, md5s=md5s)
    return SourceRecord(
        name=SOURCE,
        status="ok",
        files=retain_existing(SOURCE, manifest, raw_root, files),
        note=(
            f"tier={'full' if full else 'quick'}; Zenodo record {RECORD_ID}; "
            "raw .h5 recordings (~60 GB) excluded by design; "
            "diazepam series control/3/10/30/50 uM in 4 organoids"
        ),
        **META,
    )

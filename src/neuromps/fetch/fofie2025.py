"""Fofie et al. 2025: hiPSC-derived nociceptors on Axion MEA, concentration-response tables.

Compound overlap with the rest of the asset (TTX, GABA, lidocaine, DNQX,
retigabine and others) in a single 1.25 MB workbook of per-well features.
These are sensory neurons, not cortical: use for mechanism-level comparison
only, never as a cortical human reference.
"""

from pathlib import Path

from neuromps.fetch.common import fetch_many, retain_existing
from neuromps.fetch.download import SourceBlocked, get_json
from neuromps.fetch.manifest import Manifest, SourceRecord

SOURCE = "fofie2025"
ARTICLE_ID = 28755155
API = "https://api.figshare.com/v2/articles"
EXPECTED_LICENSE = "CC0"
META = {
    "landing_url": "https://doi.org/10.6084/m9.figshare.28755155.v1",
    "license": "CC0-1.0",
    "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
    "citation": (
        "Fofie E, Granja-Vazquez R, Truong V, Walsh T, Price TJ, Biswas S, Dussor G, Pancrazio JJ, "
        "Kolber BJ. Raw data, Cell Reports Methods 2025. doi:10.6084/m9.figshare.28755155.v1"
    ),
}


def fetch(raw_root: Path, manifest: Manifest, full: bool) -> SourceRecord:
    """Download the concentration-response workbook into ``data/raw/fofie2025/``."""
    art = get_json(f"{API}/{ARTICLE_ID}")
    got = str(art.get("license", {}).get("name", ""))
    if got != EXPECTED_LICENSE:
        raise SourceBlocked(f"license changed upstream: expected {EXPECTED_LICENSE!r}, got {got!r}")
    cached = manifest.file_index()
    files = []
    for f in art["files"]:
        rel = f"{SOURCE}/{f['name'].replace(' ', '_')}"
        md5 = f.get("computed_md5") or None
        files += fetch_many([(f["download_url"], rel)], raw_root, cached, md5s={rel: md5})
    if not files:
        raise SourceBlocked(f"no files in figshare article {ARTICLE_ID}")
    return SourceRecord(
        name=SOURCE,
        status="ok",
        files=retain_existing(SOURCE, manifest, raw_root, files),
        note="hiPSC sensory neurons (RealDRG), Axion Maestro Pro 48- and 96-well; features only",
        **META,
    )

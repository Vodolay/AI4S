"""Brewer et al. 2021: rat hippocampal subregions across instrumented microtunnels.

The asset's two-compartment evidence comes from devices whose microchannels
carry no electrodes, which is precisely the confound §7.7 could not rule out:
propagation direction had to be recovered from the chambers rather than read
from the connection itself. In this device the tunnels between the four
subregions are themselves over electrodes, so the crossing is measured.

Small enough to fetch on both tiers - 17 MB of spike trains against the 109 MB
and 2.5 GB archives elsewhere in the asset.
"""

from pathlib import Path

from neuromps.fetch import zenodo
from neuromps.fetch.common import fetch_many, retain_existing
from neuromps.fetch.download import SourceBlocked
from neuromps.fetch.manifest import Manifest, SourceRecord

SOURCE = "brewer2021"
RECORD_ID = 4679064
EXPECTED_LICENSE = "cc-by-4.0"
WANTED = ("readme.md", "spike_trains_data.zip")
ARCHIVE_NAME = "spike_trains_data.zip"
META = {
    "landing_url": "https://doi.org/10.5281/zenodo.4679064",
    "license": "CC-BY-4.0",
    "license_url": "https://creativecommons.org/licenses/by/4.0/",
    "citation": (
        "Vakilna YS, Tang WC, Wheeler BC, Brewer GJ. The flow of axonal information among "
        "hippocampal subregions: 1. Feedforward and feedback network spatial dynamics "
        "underpinning emergent information processing. Front. Neural Circuits (2021). "
        "Dataset doi:10.5281/zenodo.4679064"
    ),
}


def fetch(raw_root: Path, manifest: Manifest, full: bool) -> SourceRecord:
    """Download the spike trains and the readme that documents their layout."""
    rec = zenodo.record(RECORD_ID)
    got = zenodo.license_id(rec)
    if got != EXPECTED_LICENSE:
        raise SourceBlocked(f"license changed upstream: expected {EXPECTED_LICENSE!r}, got {got!r}")
    available = zenodo.files(rec)
    missing = [name for name in WANTED if name not in available]
    if missing:
        raise SourceBlocked(f"files missing from record {RECORD_ID}: {missing}")
    cached = manifest.file_index()
    items = [(available[name].url, f"{SOURCE}/{name}") for name in WANTED]
    md5s = {f"{SOURCE}/{name}": available[name].md5 for name in WANTED}
    files = fetch_many(items, raw_root, cached, max_bytes=1 << 30, md5s=md5s)
    return SourceRecord(
        name=SOURCE,
        status="ok",
        files=retain_existing(SOURCE, manifest, raw_root, files),
        note=(
            f"tier={'full' if full else 'quick'}; Zenodo record {RECORD_ID}; "
            "10 cultures, four hippocampal subregions each over a 60-electrode MCS array, "
            "300 s spontaneous at 25 kHz; the analysis scripts in the record are not fetched"
        ),
        **META,
    )

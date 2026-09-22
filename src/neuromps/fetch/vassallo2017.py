"""Vassallo et al. 2017: the EPA arm of the multi-laboratory MEA study.

The source this asset most needed. Every other multi-compound set here is
fitted curves or per-well summaries; this one is spike times, six compounds in
a concentration-response design, and - the part that matters for every model in
`models/` - a **baseline recorded on the same chip** before the first dose.

The archive is a single 199 MB zip from EPA ScienceHub. Inside, 38 files hold
one chip's spikes as `sample_index<TAB>channel` at 25 kHz, continuously across
the whole session, and `Logfile_EPAv1.xls` says which seconds belong to the
baseline and to each concentration. Without that logfile the spike files cannot
be cut into records at all: the session runs for hours and nothing in the spike
times marks where a dose went in.

Licence: U.S. Public Domain, 17 U.S.C. section 105, declared on the EPA
ScienceHub licence page, which this fetcher re-reads on every run.
"""

from pathlib import Path

import requests

from neuromps.fetch.common import fetch_many, retain_existing
from neuromps.fetch.download import USER_AGENT, SourceBlocked
from neuromps.fetch.manifest import Manifest, SourceRecord

SOURCE = "vassallo2017"
ARCHIVE_NAME = "Vassallo Data.zip"
ARCHIVE_URL = "https://pasteur.epa.gov/uploads/10.23719/1517583/Vassallo%20Data.zip"
LICENSE_URL = "https://pasteur.epa.gov/license/sciencehub-license.html"
# Phrases that must still be on the EPA licence page for the public-domain
# claim in META to hold. Checked rather than trusted, as for every other source.
LICENSE_PHRASES = ("public domain", "17 u.s.c")
META = {
    "landing_url": "https://catalog.data.gov/dataset/data-set-for-vassallo-et-al",
    "license": "US-Public-Domain",
    "license_url": LICENSE_URL,
    "citation": (
        "Vassallo A, Chiappalone M, De Camargos Lopes R, et al. A multi-laboratory evaluation "
        "of microelectrode array-based measurements of neural network activity for acute "
        "neurotoxicity testing. Neurotoxicology 60:280-292 (2017). Dataset: U.S. EPA Office of "
        "Research and Development, doi:10.23719/1517583 (U.S. Public Domain)."
    ),
}


def check_license(timeout: float = 60) -> str:
    """Re-read the EPA licence page; raise if it no longer grants public domain."""
    response = requests.get(LICENSE_URL, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    response.raise_for_status()
    text = response.text.lower()
    missing = [phrase for phrase in LICENSE_PHRASES if phrase not in text]
    if missing:
        raise SourceBlocked(
            f"EPA licence page no longer states {missing}; not treating this as public domain"
        )
    return LICENSE_URL


def fetch(raw_root: Path, manifest: Manifest, full: bool) -> SourceRecord:
    """Download the archive on the full tier; the quick tier skips it."""
    check_license()
    if not full:
        return SourceRecord(
            name=SOURCE,
            status="skipped",
            files=[],
            note="199 MB archive; full tier only (run: fetch.runner --full)",
            **META,
        )
    cached = manifest.file_index()
    files = fetch_many(
        [(ARCHIVE_URL, f"{SOURCE}/{ARCHIVE_NAME}")], raw_root, cached, max_bytes=1 << 30
    )
    return SourceRecord(
        name=SOURCE,
        status="ok",
        files=retain_existing(SOURCE, manifest, raw_root, files),
        note=(
            "tier=full; EPA ScienceHub doi:10.23719/1517583; 38 chips, six compounds "
            "(deltamethrin, domoic acid, chlorpyrifos-oxon, acetaminophen, glyphosate, "
            "dimethyl phthalate) in concentration-response with a same-chip baseline; "
            "spike times at 25 kHz, dose windows in Logfile_EPAv1.xls"
        ),
        **META,
    )

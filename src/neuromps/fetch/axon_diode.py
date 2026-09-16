"""Axon-diode / asymmetric-microchannel chip papers (Sorbonne lineage): design parameters only.

Only open-access CC-BY articles are fetched (full-text JATS XML). The original
axon-diode paper (Peyrin et al. 2011, Lab Chip, doi:10.1039/c1lc20014c) and the
Methods Cell Biol chapter (Courte et al. 2018, doi:10.1016/bs.mcb.2018.07.002)
are not open access and are cited only, never downloaded.
"""

from pathlib import Path

from neuromps.fetch.common import retain_existing
from neuromps.fetch.europepmc import fetch_article
from neuromps.fetch.manifest import Manifest, SourceRecord

SOURCE = "axon_diode"
ARTICLES = {
    "PMC4406441": "Renault et al. 2015 PLoS One, doi:10.1371/journal.pone.0120680",
    "PMC6265304": "Lassus et al. 2018 Sci Rep, doi:10.1038/s41598-018-35802-9",
    "PMC7078319": "Courte et al. 2020 Sci Rep, doi:10.1038/s41598-020-61757-x",
}
NOT_OPEN = (
    "Peyrin 2011 Lab Chip doi:10.1039/c1lc20014c",
    "Courte 2018 Methods Cell Biol doi:10.1016/bs.mcb.2018.07.002",
)
META = {
    "landing_url": "https://europepmc.org/",
    "license": "CC-BY-4.0 (per article, verified from XML)",
    "license_url": "https://creativecommons.org/licenses/by/4.0/",
    "citation": "; ".join(ARTICLES.values()),
}


def fetch(raw_root: Path, manifest: Manifest, full: bool) -> SourceRecord:
    """Download full-text XML for each open-access article into ``data/raw/axon_diode/``."""
    cached = manifest.file_index()
    files, licenses = [], []
    for pmcid in ARTICLES:
        recs, lic = fetch_article(pmcid, raw_root, SOURCE, cached)
        files += recs
        licenses.append(f"{pmcid}: {lic}")
    note = (
        "licenses verified from XML: "
        + "; ".join(licenses)
        + " | not open access, cited only: "
        + "; ".join(NOT_OPEN)
    )
    return SourceRecord(
        name=SOURCE,
        status="ok",
        files=retain_existing(SOURCE, manifest, raw_root, files),
        note=note,
        **META,
    )

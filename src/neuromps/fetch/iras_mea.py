"""Optional: IRAS/Utrecht (Westerink lab) MEA seizure-liability pharmacology, via Europe PMC.

Tukker et al. 2020 compares hiPSC-derived neuronal co-cultures with rat primary
cortical cultures on MEA across reference compounds. Its supplementary data
(docx) holds per-compound tables. The article is CC BY-NC 4.0: non-commercial
research reuse only; do not redistribute derived tables commercially.
"""

from pathlib import Path

from neuromps.fetch.common import retain_existing
from neuromps.fetch.europepmc import fetch_article
from neuromps.fetch.manifest import Manifest, SourceRecord

SOURCE = "iras_mea"
ARTICLES = {"PMC7657345": "10.1093/toxsci/kfaa136"}
META = {
    "landing_url": "https://doi.org/10.1093/toxsci/kfaa136",
    "license": "CC-BY-NC-4.0",
    "license_url": "https://creativecommons.org/licenses/by-nc/4.0/",
    "citation": (
        "Tukker AM, Wijnolts FMJ, de Groot A, Westerink RHS. Applicability of hiPSC-derived "
        "neuronal cocultures and rodent primary cortical cultures for in vitro seizure "
        "liability assessment. Toxicol Sci 178(1):71-87 (2020). doi:10.1093/toxsci/kfaa136"
    ),
}


def fetch(raw_root: Path, manifest: Manifest, full: bool) -> SourceRecord:
    """Download article XML + supplementary zip into ``data/raw/iras_mea/``."""
    cached = manifest.file_index()
    files, licenses = [], []
    for pmcid in ARTICLES:
        recs, lic = fetch_article(pmcid, raw_root, SOURCE, cached, supplementary=True)
        files += recs
        licenses.append(f"{pmcid}: {lic}")
    return SourceRecord(
        name=SOURCE,
        status="ok",
        files=retain_existing(SOURCE, manifest, raw_root, files),
        note="license verified from article XML: " + "; ".join(licenses),
        **META,
    )

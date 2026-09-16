"""Kapucu et al. 2022 (Sci Data 9:120): hPSC-derived vs rat cortical MEA dataset on G-Node GIN.

Only processed files are fetched (spike times, expLogs, meaRtools per-well
features). The raw 12.5 kHz .h5 recordings (~2.4 TB) are deliberately excluded.
"""

from pathlib import Path

from neuromps.fetch.common import fetch_many, retain_existing
from neuromps.fetch.europepmc import fetch_article
from neuromps.fetch.manifest import Manifest, SourceRecord

SOURCE = "kapucu2022"
COMMIT = "0f0f13517bf6cfc382b5f3c3f9219f13d76d798a"  # pinned GIN commit (master, 2026-09-16)
BASE = f"https://gin.g-node.org/NeuroGroup_TUNI/Comparative_MEA_dataset/raw/{COMMIT}"
PMCID = "PMC8969177"
META = {
    "landing_url": "https://doi.org/10.12751/g-node.wvr3jf",
    "license": "CC-BY-4.0 (Data/); article CC-BY-4.0",
    "license_url": "https://creativecommons.org/licenses/by/4.0/",
    "citation": (
        "Kapucu FE, Vinogradov A, Hyvärinen T, Ylä-Outinen L, Narkilahti S. Comparative "
        "microelectrode array data of the functional development of hPSC-derived and rat "
        "neuronal networks. Sci Data 9, 120 (2022). doi:10.1038/s41597-022-01242-4"
    ),
}
_LIST = Path(__file__).with_name("kapucu2022_files.tsv")


def file_list(full: bool) -> list[str]:
    """Repository paths to fetch; ``full`` adds development/PCA spike-time CSVs."""
    rows = [ln.split("\t") for ln in _LIST.read_text().splitlines() if not ln.startswith("#")]
    return [p for p, tier, _ in rows[1:] if full or tier == "quick"]


def fetch(raw_root: Path, manifest: Manifest, full: bool) -> SourceRecord:
    """Download the selected tier into ``data/raw/kapucu2022/``."""
    cached = manifest.file_index()
    items = [(f"{BASE}/{p}", f"{SOURCE}/{p}") for p in file_list(full)]
    files = fetch_many(items, raw_root, cached)
    paper, _ = fetch_article(PMCID, raw_root, f"{SOURCE}/paper", cached)
    tier = "full" if full else "quick"
    return SourceRecord(
        name=SOURCE,
        status="ok",
        files=retain_existing(SOURCE, manifest, raw_root, files + paper),
        note=f"tier={tier}; GIN commit {COMMIT}; raw .h5 excluded by design",
        **META,
    )

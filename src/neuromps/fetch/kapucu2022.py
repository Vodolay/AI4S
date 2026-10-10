"""Kapucu et al. 2022 (Sci Data 9:120): hPSC-derived vs rat cortical MEA dataset on G-Node GIN.

Only processed files are fetched (spike times, expLogs, meaRtools per-well
features). The raw 12.5 kHz .h5 recordings (~2.4 TB) are deliberately excluded.
"""

import logging
from pathlib import Path

import requests

from neuromps.fetch import mirror
from neuromps.fetch.common import fetch_many, retain_existing
from neuromps.fetch.download import USER_AGENT
from neuromps.fetch.europepmc import fetch_article
from neuromps.fetch.manifest import Manifest, SourceRecord

log = logging.getLogger("neuromps.fetch")

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

# GIN is the only host, and it went unreachable in the middle of the first
# fresh-clone full run. The quick tier - which every run, and CI, needs - is
# mirrored as a release asset of this repository: the same bytes as GIN serves
# at COMMIT, so the manifest's hashes hold for either copy. Built by
# `python -m neuromps.fetch.mirror`; the hash pins exactly that archive.
MIRROR_TAG = "data-mirror-kapucu2022-quick"
MIRROR_URL = (
    f"https://github.com/Vodolay/AI4S/releases/download/{MIRROR_TAG}/"
    f"kapucu2022_quick_{COMMIT[:8]}.zip"
)
MIRROR_SHA256 = "6e1b7d45ef1d21a53d2e95c94bd5482b7127c65c868510e256e028355018a686"
MIRROR_README = f"""Kapucu et al. 2022, quick tier of the NeuroMPS pipeline - a mirror.

These files are unmodified copies of the processed data in G-Node GIN repository
NeuroGroup_TUNI/Comparative_MEA_dataset at commit {COMMIT}:
  {BASE}
Dataset DOI: {META["landing_url"]}

License: Creative Commons Attribution 4.0 International (CC BY 4.0),
{META["license_url"]}. The authors did not prepare or endorse this copy.

Please cite:
  {META["citation"]}

The NeuroMPS fetcher uses this archive only when GIN cannot be reached, and only
after its SHA-256 matches the value pinned in src/neuromps/fetch/kapucu2022.py.
Every file's SHA-256 is also recorded in data/raw/MANIFEST.json of the repository
https://github.com/Vodolay/AI4S.
"""


def file_list(full: bool) -> list[str]:
    """Repository paths to fetch; ``full`` adds development/PCA spike-time CSVs."""
    rows = [ln.split("\t") for ln in _LIST.read_text().splitlines() if not ln.startswith("#")]
    return [p for p, tier, _ in rows[1:] if full or tier == "quick"]


def gin_reachable(timeout: float = 20, attempts: int = 2) -> bool:
    """Whether GIN answers for one small file of the pinned commit, decided in seconds.

    In the outage that prompted the mirror, connections hung until their timeout,
    so trying 169 files with retries before falling back would have taken hours.
    """
    for _ in range(attempts):
        try:
            response = requests.get(
                f"{BASE}/datacite.yml", timeout=timeout, headers={"User-Agent": USER_AGENT}
            )
            if response.status_code < 400:
                return True
        except requests.RequestException:
            pass
    return False


def fetch(raw_root: Path, manifest: Manifest, full: bool) -> SourceRecord:
    """Download the selected tier into ``data/raw/kapucu2022/``."""
    cached = manifest.file_index()
    items = [(f"{BASE}/{p}", f"{SOURCE}/{p}") for p in file_list(full)]
    origin = f"GIN commit {COMMIT}"
    try:
        if not full and not gin_reachable():
            raise requests.ConnectionError(f"{BASE} did not answer")
        files = fetch_many(items, raw_root, cached)
    except requests.RequestException as exc:
        if full:
            raise  # the mirror holds the quick tier only
        log.warning("[%s] GIN failed (%s); using the checksummed mirror", SOURCE, exc)
        files = mirror.fetch_members(
            MIRROR_URL, MIRROR_SHA256, raw_root, [rel for _, rel in items], cached
        )
        origin = f"GIN unreachable; identical files from the project mirror of GIN commit {COMMIT}"
    paper, _ = fetch_article(PMCID, raw_root, f"{SOURCE}/paper", cached)
    tier = "full" if full else "quick"
    return SourceRecord(
        name=SOURCE,
        status="ok",
        files=retain_existing(SOURCE, manifest, raw_root, files + paper),
        note=f"tier={tier}; {origin}; raw .h5 excluded by design",
        **META,
    )

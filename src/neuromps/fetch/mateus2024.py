"""Mateus et al. 2024: hippocampal circuits on asymmetric-microchannel MEAs.

The closest published analogue of the two-compartment chip the twin models:
straight (``Control``) microchannels against three asymmetric geometries —
``Arrows``, ``Rams`` and ``Tesla`` valves — with the geometry known by design.
That makes it the real-data counterpart to the simulated directionality
experiment, where until now the ground truth existed only in the simulator.

**The licence needs stating carefully, because the source contradicts itself.**
The Zenodo record's licence field says ``cc-by-4.0`` and the record is open
access; the README shipped inside the deposit says, in capitals, CC BY-NC-ND.
Either reading is checkable by anyone, and they cannot both govern. On
2026-09-21 the corresponding author, Paulo Aguiar, wrote to us: *"Please
consider CC BY 4.0 only, as stated in the metadata. (Unfortunately, CC BY-NC-ND
was the default in the readme and was unintentionally left there - we cannot
change it now)."*

This module therefore does three things rather than quietly preferring the
convenient reading: it blocks if the *metadata* licence ever moves off
``cc-by-4.0``, it reads the README on every fetch and records whatever licence
claim it finds, and it writes the author's clarification into the manifest so
the contradiction and its resolution travel with the data.
"""

import re
from pathlib import Path

from neuromps.fetch import zenodo
from neuromps.fetch.common import fetch_many, retain_existing
from neuromps.fetch.download import SourceBlocked
from neuromps.fetch.manifest import Manifest, SourceRecord

SOURCE = "mateus2024"
RECORD_ID = 14525182
EXPECTED_LICENSE = "cc-by-4.0"
README = "README.txt"
QUICK_FILES = (README,)
FULL_FILES = ("1_MEA_Recordings.zip",)
# 2_Raw_Images.zip (66 MB of widefield microscopy) is not fetched: the asset is
# built from spike data, and downloading images we will not use is not free.

# Paraphrased rather than quoted: the clarification arrived as private
# correspondence, and the authors agreed to a licence reading, not to having
# their message published. The date and the substance are what a reader needs
# to judge the decision; the wording is theirs to release.
CLARIFICATION = (
    "Licence conflict resolved by the deposit's corresponding author by email on 2026-09-21, "
    "copying the first author: they confirmed that the Zenodo metadata licence (CC BY 4.0) is "
    "the one they intend, and that the README's CC BY-NC-ND line was an unintentional default "
    "they can no longer edit. The metadata licence field is the operative licence and is "
    "publicly checkable; the README's contradicting text is recorded below rather than ignored. "
    "The correspondence itself is private and a reader cannot verify it - see data/SOURCES.md."
)

NON_FREE_TERMS = re.compile(r"BY-NC-ND|NON-?COMMERCIAL|NODERIV", re.IGNORECASE)

META = {
    "landing_url": "https://doi.org/10.5281/zenodo.14525182",
    "license": "CC-BY-4.0",
    "license_url": "https://creativecommons.org/licenses/by/4.0/",
    "citation": (
        "Mateus J, Melo P, Aroso M, Charlot B, Aguiar P. Influence of asymmetric microchannels "
        "in the structure and function of engineered neuronal circuits. bioRxiv 2024, "
        "doi:10.1101/2024.07.09.602729. Dataset: doi:10.5281/zenodo.14525182 (CC BY 4.0)."
    ),
}


def readme_license_claim(text: str) -> str | None:
    """The README's own licence sentence, when it claims a more restrictive licence.

    Returned so the manifest can record what the deposit says about itself. A
    fetcher that simply trusted the metadata would leave no trace that the two
    disagree, and the disagreement is exactly what a reader needs to see.
    """
    for line in text.splitlines():
        if NON_FREE_TERMS.search(line):
            return " ".join(line.split())[:300]
    return None


def license_note(readme_text: str | None) -> str:
    """How the licence stands on this fetch: metadata, README, and the resolution."""
    if readme_text is None:
        return f"{CLARIFICATION} README not read on this run."
    claim = readme_license_claim(readme_text)
    if claim is None:
        return (
            f"{CLARIFICATION} On this fetch the README no longer states a restrictive licence, "
            "so the deposit is now self-consistent."
        )
    return f"{CLARIFICATION} README as fetched still states: '{claim}'"


def fetch(raw_root: Path, manifest: Manifest, full: bool) -> SourceRecord:
    """Download the README always, and the spike-data archive when ``full``."""
    rec = zenodo.record(RECORD_ID)
    got = zenodo.license_id(rec)
    if got != EXPECTED_LICENSE:
        raise SourceBlocked(
            f"license changed upstream: expected {EXPECTED_LICENSE!r}, got {got!r}. "
            "The inclusion of this source rests on the metadata licence; do not fetch it "
            "under a different one without checking with the depositor again."
        )
    available = zenodo.files(rec)
    wanted = QUICK_FILES + (FULL_FILES if full else ())
    missing = [n for n in wanted if n not in available]
    if missing:
        raise SourceBlocked(f"files missing from record {RECORD_ID}: {missing}")
    cached = manifest.file_index()
    items = [(available[n].url, f"{SOURCE}/{n}") for n in wanted]
    md5s = {f"{SOURCE}/{n}": available[n].md5 for n in wanted}
    files = fetch_many(items, raw_root, cached, max_bytes=2 << 30, md5s=md5s)
    readme_path = raw_root / SOURCE / README
    readme_text = readme_path.read_text(errors="replace") if readme_path.exists() else None
    return SourceRecord(
        name=SOURCE,
        status="ok",
        files=retain_existing(SOURCE, manifest, raw_root, files),
        note=(
            f"tier={'full' if full else 'quick'}; Zenodo record {RECORD_ID}; "
            "hippocampal cultures on MEA2100-256, 10 min spontaneous recordings, DIV 12-21; "
            "straight (Control) against asymmetric microchannels (Arrows, Rams, Tesla); "
            "2_Raw_Images.zip (66 MB widefield images) not fetched. " + license_note(readme_text)
        ),
        **META,
    )

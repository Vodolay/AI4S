"""The FlyWire v783 whole adult fly brain, for the chip-to-animal study.

Three licensed pieces, about 440 MB in all:

- the connectome-constrained leaky integrate-and-fire model of Shiu et al. 2024
  (MIT), read at one pinned commit: its neuron list, its signed connectivity
  table derived from FlyWire 783, and its license text; plus the same two
  tables for FlyWire 630, the release the paper used, for a replication check;
- per-neuron synapse counts by neuropil, from the FlyWire connectivity release
  (Zenodo 10676866, CC-BY-4.0), which place each neuron in a brain region;
- the neuron annotations of Schlegel et al. 2024 (Zenodo 23037340, CC-BY-4.0),
  which name the cell types and give each neuron's predicted transmitter.

The GitHub copy of the annotations carries no license, so only the Zenodo
archive is used. This source is optional: a default fetch leaves it alone,
and ``--only flywire783`` (or the fly stage) asks for it.
"""

from pathlib import Path

from neuromps.fetch import zenodo
from neuromps.fetch.common import fetch_many, retain_existing
from neuromps.fetch.download import SourceBlocked
from neuromps.fetch.manifest import Manifest, SourceRecord

SOURCE = "flywire783"
SHIU_COMMIT = "91bdd1e7dcf193f3e7ca5a8933497fcef63b7960"
SHIU_RAW = f"https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/{SHIU_COMMIT}"
SHIU_FILES = ("Completeness_783.csv", "Connectivity_783.parquet", "LICENSE")
# The release the paper itself used. Fetched only to check that the model gives
# the published result on 630 and a matching one on 783; nothing else reads it.
SHIU_630_FILES = (
    "2023_03_23_completeness_630_final.csv",
    "2023_03_23_connectivity_630_final.parquet",
)
CONNECTIVITY_RECORD = 10676866
CONNECTIVITY_FILES = ("per_neuron_neuropil_count_post_783.feather",)
ANNOTATION_RECORD = 23037340
ANNOTATION_ZIP = "flyconnectome/flywire_annotations-v3.2.0.zip"
ANNOTATION_MEMBER = "supplemental_files/Supplemental_file1_neuron_annotations.tsv"
EXPECTED_LICENSE = "cc-by-4.0"
META = {
    "landing_url": "https://doi.org/10.5281/zenodo.10676866",
    "license": "CC-BY-4.0 (FlyWire data, annotations); MIT (Shiu et al. model files)",
    "license_url": "https://creativecommons.org/licenses/by/4.0/",
    "citation": (
        "Dorkenwald S, et al. Neuronal wiring diagram of an adult brain. Nature 634, 124-138 "
        "(2024); FlyWire connectivity data, doi:10.5281/zenodo.10676866. Schlegel P, et al. "
        "Whole-brain annotation and multi-connectome cell typing of Drosophila. Nature 634, "
        "139-152 (2024); annotations v3.2.0, doi:10.5281/zenodo.23037340. Shiu PK, et al. A "
        "Drosophila computational brain model reveals sensorimotor processing. Nature 634, "
        "210-219 (2024); code github.com/philshiu/Drosophila_brain_model"
    ),
}


def local_name(zenodo_key: str) -> str:
    """Where a Zenodo file lands under ``data/raw/flywire783/``."""
    return f"{SOURCE}/{zenodo_key.rsplit('/', 1)[-1]}"


def _zenodo_items(record_id: int, names: tuple[str, ...]) -> tuple[list, dict]:
    rec = zenodo.record(record_id)
    got = zenodo.license_id(rec)
    if got != EXPECTED_LICENSE:
        raise SourceBlocked(
            f"license changed upstream on record {record_id}: expected {EXPECTED_LICENSE!r}, "
            f"got {got!r}"
        )
    available = zenodo.files(rec)
    missing = [n for n in names if n not in available]
    if missing:
        raise SourceBlocked(f"files missing from record {record_id}: {missing}")
    items = [(available[n].url, local_name(n)) for n in names]
    return items, {local_name(n): available[n].md5 for n in names}


def fetch(raw_root: Path, manifest: Manifest, full: bool) -> SourceRecord:
    """Download the model files, the neuropil counts and the annotation archive."""
    connectivity, md5s = _zenodo_items(CONNECTIVITY_RECORD, CONNECTIVITY_FILES)
    annotations, more = _zenodo_items(ANNOTATION_RECORD, (ANNOTATION_ZIP,))
    model = [(f"{SHIU_RAW}/{name}", f"{SOURCE}/{name}") for name in SHIU_FILES + SHIU_630_FILES]
    files = fetch_many(
        model + connectivity + annotations,
        raw_root,
        manifest.file_index(),
        max_bytes=1 << 30,
        md5s={**md5s, **more},
    )
    return SourceRecord(
        name=SOURCE,
        status="ok",
        files=retain_existing(SOURCE, manifest, raw_root, files),
        note=(
            f"optional (fly stage only); Shiu et al. model files at commit {SHIU_COMMIT[:12]}; "
            f"Zenodo {CONNECTIVITY_RECORD} neuropil counts; Zenodo {ANNOTATION_RECORD} "
            "annotations archive (the GitHub copy has no license and is not used)"
        ),
        **META,
    )

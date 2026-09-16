"""US EPA ToxCast invitrodb v4.3: CCTE_Shafer MEA acute and network-formation (dev) assays.

Quick tier: assay annotations, cytotoxicity/target tables, release docs and the
MEA assay description PDFs. Full tier adds the all-assay mc5-6 hit-call CSV
(2.55 GB uncompressed), extracted from the 7.5 GB summary zip via HTTP ranges.
Filtering to MEA AEIDs happens later in ETL, so the raw file stays unmodified.
"""

import logging
from pathlib import Path

from neuromps.fetch.common import fetch_many, retain_existing
from neuromps.fetch.download import get_json
from neuromps.fetch.manifest import Manifest, SourceRecord
from neuromps.fetch.remote_zip import extract_member

SOURCE = "toxcast_mea"
CLOWDER = "https://clowder.edap-cluster.com/api"
SUMMARY_DATASET = "687e392ae4b02565bc3e2914"
RELEASE_DATASET = "687e3921e4b02565bc3e290f"
ASSAY_DOCS_DATASET = "6894f2dae4b025654d12b716"
SUMMARY_ZIP = "INVITRODB_SUMMARY.zip"
MC56_MEMBER = "mc5-6_winning_model_fits-flags_invitrodbv4_3_AUG2024.csv"
QUICK_SUMMARY_FILES = (
    "assay_annotations_invitrodb_v4_3_AUG2025.xlsx",
    "assay_target_mappings_invitrodb_v4_3_AUG2025.xlsx",
    "cytotox_invitrodb_v4_3_AUG2025.xlsx",
    "DB_release_README_SUMMARY.pdf",
)
META = {
    "landing_url": "https://doi.org/10.23645/epacomptox.6062623.v14",
    "license": "CC0-1.0",
    "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
    "citation": (
        "U.S. EPA Center for Computational Toxicology and Exposure. ToxCast Database: "
        "invitrodb version 4.3. doi:10.23645/epacomptox.6062623.v14"
    ),
}
log = logging.getLogger("neuromps.fetch")


def _files(dataset: str) -> dict[str, str]:
    """Map filename -> Clowder file id for a dataset."""
    return {f["filename"]: f["id"] for f in get_json(f"{CLOWDER}/datasets/{dataset}/files")}


def _blob(file_id: str) -> str:
    return f"{CLOWDER}/files/{file_id}/blob"


def quick_items() -> list[tuple[str, str]]:
    """(url, rel_path) for annotation tables, release notes and MEA assay PDFs."""
    summary, release = _files(SUMMARY_DATASET), _files(RELEASE_DATASET)
    items = [(_blob(summary[n]), f"{SOURCE}/summary/{n}") for n in QUICK_SUMMARY_FILES]
    items += [(_blob(i), f"{SOURCE}/release/{n}") for n, i in release.items()]
    docs = _files(ASSAY_DOCS_DATASET)
    mea = sorted(n for n in docs if "CCTE_Shafer_MEA_" in n)
    if not mea:
        raise RuntimeError("no CCTE_Shafer_MEA assay description documents found")
    items += [(_blob(docs[n]), f"{SOURCE}/assay_docs/{n}") for n in mea]
    return items


def fetch(raw_root: Path, manifest: Manifest, full: bool) -> SourceRecord:
    """Download into ``data/raw/toxcast_mea/``."""
    cached = manifest.file_index()
    files = fetch_many(quick_items(), raw_root, cached)
    if full:
        rel = f"{SOURCE}/summary/{MC56_MEMBER}"
        if rel in cached and (raw_root / rel).exists():
            files.append(cached[rel])  # hash re-verified in retain_existing on later runs
        else:
            zip_url = _blob(_files(SUMMARY_DATASET)[SUMMARY_ZIP])
            log.info("  extracting %s from remote %s (~661 MB transfer)", MC56_MEMBER, SUMMARY_ZIP)
            files.append(extract_member(zip_url, MC56_MEMBER, raw_root, rel))
    return SourceRecord(
        name=SOURCE,
        status="ok",
        files=retain_existing(SOURCE, manifest, raw_root, files),
        note=f"tier={'full' if full else 'quick'}; invitrodb v4.3 via EPA Clowder",
        **META,
    )

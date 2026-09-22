"""Build the data asset: raw files -> validated schema records -> parquet + JSONL + provenance.

Run: ``python -m neuromps.etl.build`` (or ``python run_all.py --stage data``).

Nothing is written unless every record validates against `schema/neuromps_schema.json`.
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import pandas as pd

from neuromps.etl import kapucu, screenneuropharm, sharf, trujillo
from neuromps.etl.toxcast_mea import (
    INTERIM_PARQUET,
    MC56_CSV,
    build_curve_records,
    load_mc56,
)
from neuromps.fetch.manifest import Manifest, utc_now
from neuromps.metrics import update_metrics
from neuromps.paths import interim_dir, manifest_path, processed_dir
from neuromps.qc.flags import add_cross_record_flags
from neuromps.qc.schema import check_traceable, validate_records

log = logging.getLogger("neuromps.etl")
REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = REPO_ROOT / "data" / "raw"

NESTED_ONLY = ("provenance",)
COMPARTMENT_CACHE = REPO_ROOT / "data" / "interim" / "screenneuropharm_compartments.json"


def manifest_hashes(raw_root: Path = RAW_ROOT) -> dict[str, str]:
    """Map file name -> SHA256 from data/raw/MANIFEST.json."""
    manifest = Manifest.load(manifest_path(raw_root))
    return {Path(rec.path).name: rec.sha256 for rec in manifest.file_index().values()}


def flatten_records(records: list[dict]) -> pd.DataFrame:
    """One row per record, nested blocks as dotted columns; provenance stays in the JSONL."""
    flat = pd.json_normalize(
        [{k: v for k, v in r.items() if k not in NESTED_ONLY} for r in records]
    )
    for column in flat.columns:
        if flat[column].map(lambda v: isinstance(v, list)).any():
            flat[column] = flat[column].map(_flatten_list)
    return flat


def _flatten_list(value: object) -> object:
    """Lists of strings become 'a;b'; structured lists (coupling) become JSON."""
    if not isinstance(value, list):
        return value
    if all(isinstance(v, str) for v in value):
        return ";".join(value)
    return json.dumps(value, default=str)


def collect_records(raw_root: Path = RAW_ROOT, full: bool = False) -> list[dict]:
    """Build records from every source that has been fetched."""
    hashes = manifest_hashes(raw_root)
    records: list[dict] = []
    for plate in kapucu.PHARMACOLOGY_PLATES:
        plate_dir = raw_root / "kapucu2022" / "Data" / plate.directory
        if not plate_dir.exists():
            log.warning("kapucu plate missing, skipping: %s", plate.directory)
            continue
        built = kapucu.build_plate_records(plate_dir, plate, hashes, raw_root=raw_root)
        log.info("kapucu %s: %d records", plate.directory, len(built))
        records += built
    records += _toxcast_records(hashes, full)
    records += _screenneuropharm_records(hashes, raw_root, full)
    records += _archive_records(sharf, hashes, raw_root, full)
    records += _mateus_records(hashes, raw_root)
    records += _trujillo_records(hashes, raw_root, full)
    add_cross_record_flags(records)  # replicate divergence, outliers, unpaired exposures
    return records


def _mateus_records(hashes: dict[str, str], raw_root: Path) -> list[dict]:
    """Two chamber records per Mateus recording; absent unless the archive was fetched.

    The archive is a full-tier download, so a quick run simply has no records
    from this source rather than a partial version of them.
    """
    from neuromps.etl.mateus2024_records import build_records
    from neuromps.paths import interim_dir

    archive = raw_root / "mateus2024" / "1_MEA_Recordings.zip"
    if not archive.exists():
        log.info("mateus2024 archive not fetched; skipping")
        return []
    built = build_records(
        archive,
        sha=hashes.get(archive.name),
        cache_path=interim_dir() / "mateus2024_records.json",
    )
    log.info("mateus2024: %d records", len(built))
    return built


def _trujillo_records(hashes: dict[str, str], raw_root: Path, full: bool) -> list[dict]:
    """Human organoid records, cached: feature extraction costs about 90 s.

    The spike tables are small, so unlike the other large sources this one is
    present on both tiers; a quick run simply has fewer sessions to read.
    """
    spikes_dir = raw_root / trujillo.SOURCE
    if not any(spikes_dir.glob("*.spikes.parquet")):
        log.info("trujillo2019 sessions not fetched; skipping")
        return []
    cache = interim_dir() / "trujillo2019_records.json"
    if cache.exists() and not full:
        built = json.loads(cache.read_text())
    else:
        built = trujillo.build_records(spikes_dir, hashes)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(built, default=str))
    log.info("trujillo2019: %d records", len(built))
    return built


def _toxcast_records(hashes: dict[str, str], full: bool) -> list[dict]:
    """ToxCast fitted-curve records, preferring the cached interim parquet."""
    if INTERIM_PARQUET.exists():
        df = pd.read_parquet(INTERIM_PARQUET)
    elif MC56_CSV.exists() and full:
        df = load_mc56(MC56_CSV)
    else:
        log.warning("toxcast mc5-6 absent, skipping (run: fetch.runner --full)")
        return []
    built = build_curve_records(df, hashes, source_file=MC56_CSV.name)
    log.info("toxcast: %d records", len(built))
    return built


def _screenneuropharm_records(hashes: dict[str, str], raw_root: Path, full: bool) -> list[dict]:
    """Bicuculline dose-response records; features are cached (~1 s per recording)."""
    archive = raw_root / screenneuropharm.SOURCE / screenneuropharm.ARCHIVE_NAME
    if not archive.exists():
        log.warning("screenneuropharm archive absent, skipping (run: fetch.runner --full)")
        return []
    built = screenneuropharm.build_records(
        archive, hashes, cache_path=screenneuropharm.CACHE, rebuild=full
    )
    log.info("screenneuropharm single-chamber: %d records", len(built))
    compartments = _cached_compartments(archive, hashes, full)
    log.info("screenneuropharm compartments: %d records", len(compartments))
    return built + compartments


def _cached_compartments(archive: Path, hashes: dict[str, str], full: bool) -> list[dict]:
    """Compartment records, cached: extraction plus coupling costs several minutes."""
    if COMPARTMENT_CACHE.exists() and not full:
        return json.loads(COMPARTMENT_CACHE.read_text())
    records = screenneuropharm.build_compartment_records(archive, hashes)
    COMPARTMENT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    COMPARTMENT_CACHE.write_text(json.dumps(records, default=str))
    return records


def _archive_records(module, hashes: dict[str, str], raw_root: Path, full: bool) -> list[dict]:
    """Records from an archive-backed source with a feature cache."""
    archive = raw_root / module.SOURCE / module.ARCHIVE_NAME
    if not archive.exists():
        log.warning("%s archive absent, skipping (run: fetch.runner --full)", module.SOURCE)
        return []
    built = module.build_records(archive, hashes, cache_path=module.CACHE, rebuild=full)
    log.info("%s: %d records", module.SOURCE, len(built))
    return built


def build_asset(
    records: list[dict],
    out_dir: Path | None = None,
    check_hashes: bool = True,
    manifest_path: Path | None = None,
) -> dict:
    """Validate then write `neuromps.parquet`, `neuromps_records.jsonl` and `provenance.json`.

    Records must pass the schema and, with ``check_hashes``, conformance level 2:
    every provenance hash matches `data/raw/MANIFEST.json`.
    """
    errors = {k: v for k, v in validate_records(records).items() if v}
    if not errors and check_hashes:
        errors = {r["record_id"]: e for r in records if (e := check_traceable(r, manifest_path))}
        if errors:
            raise ValueError(
                f"{len(errors)}/{len(records)} records are not traceable to MANIFEST.json "
                f"(conformance level 2), e.g. {list(errors.items())[:2]}"
            )
    if errors:
        sample = list(errors.items())[:3]
        raise ValueError(
            f"{len(errors)}/{len(records)} records failed schema validation, e.g. {sample}"
        )
    out_dir = Path(out_dir) if out_dir is not None else processed_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    flat = flatten_records(records)
    flat.to_parquet(out_dir / "neuromps.parquet", index=False)
    with (out_dir / "neuromps_records.jsonl").open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, default=str) + "\n")
    summary = {
        "n_records": len(records),
        "generated_at": utc_now(),
        "schema": "schema/neuromps_schema.json",
        "counts": _counts(records),
    }
    (out_dir / "provenance.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _counts(records: list[dict]) -> dict:
    """Data-card counts: records per source, level, species and mechanism class."""

    def tally(key) -> dict:
        return dict(sorted(Counter(key(r) for r in records).items()))

    return {
        "by_source": tally(lambda r: r["source"]),
        "by_data_level": tally(lambda r: r["data_level"]),
        "by_species": tally(lambda r: r["cell_source"]["species"]),
        "by_mechanism_class": tally(lambda r: r["compound"].get("mechanism_class") or "none"),
        "by_usable": tally(lambda r: str(r.get("qc", {}).get("usable"))),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the data stage."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--full", action="store_true", help="rebuild ToxCast records from the raw CSV")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    records = collect_records(full=args.full)
    if not records:
        raise SystemExit("no records built; run: python -m neuromps.fetch.runner --full")
    summary = build_asset(records)
    update_metrics("data_asset", summary)
    from neuromps.qc.data_card import main as write_card

    write_card([])
    log.info("wrote %s records to %s", summary["n_records"], processed_dir())
    print(json.dumps(summary["counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

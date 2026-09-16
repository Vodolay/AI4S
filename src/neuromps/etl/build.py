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

from neuromps.etl import kapucu
from neuromps.etl.toxcast_mea import (
    INTERIM_PARQUET,
    MC56_CSV,
    build_curve_records,
    load_mc56,
)
from neuromps.fetch.manifest import Manifest, utc_now
from neuromps.metrics import update_metrics
from neuromps.qc.schema import check_traceable, validate_records

log = logging.getLogger("neuromps.etl")
REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
NESTED_ONLY = ("provenance",)


def manifest_hashes(raw_root: Path = RAW_ROOT) -> dict[str, str]:
    """Map file name -> SHA256 from data/raw/MANIFEST.json."""
    manifest = Manifest.load(raw_root / "MANIFEST.json")
    return {Path(rec.path).name: rec.sha256 for rec in manifest.file_index().values()}


def flatten_records(records: list[dict]) -> pd.DataFrame:
    """One row per record, nested blocks as dotted columns; provenance stays in the JSONL."""
    flat = pd.json_normalize(
        [{k: v for k, v in r.items() if k not in NESTED_ONLY} for r in records]
    )
    for column in flat.columns:
        if flat[column].map(lambda v: isinstance(v, list)).any():
            flat[column] = flat[column].map(lambda v: ";".join(v) if isinstance(v, list) else v)
    return flat


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
    return records


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


def build_asset(
    records: list[dict],
    out_dir: Path = PROCESSED_DIR,
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
    out_dir = Path(out_dir)
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
    log.info("wrote %s records to %s", summary["n_records"], PROCESSED_DIR)
    print(json.dumps(summary["counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

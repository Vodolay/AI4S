"""Validate records against `schema/neuromps_schema.json`."""

import json
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

import jsonschema

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = REPO_ROOT / "schema" / "neuromps_schema.json"
EXAMPLES_DIR = REPO_ROOT / "schema" / "examples"


@lru_cache(maxsize=1)
def load_schema(path: Path = SCHEMA_PATH) -> dict:
    """Load and cache the JSON Schema document."""
    return json.loads(Path(path).read_text())


@lru_cache(maxsize=1)
def _validator() -> jsonschema.protocols.Validator:
    schema = load_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())


def validate_record(record: dict) -> list[str]:
    """Return human-readable validation errors for one record ([] if valid)."""
    errors = []
    for err in sorted(_validator().iter_errors(record), key=lambda e: list(e.absolute_path)):
        where = ".".join(str(p) for p in err.absolute_path) or "<record>"
        errors.append(f"{where}: {err.message}")
    return errors


def validate_records(records: Iterable[dict]) -> dict[str, list[str]]:
    """Validate many records, keyed by ``record_id`` (or ``index:<n>`` when absent)."""
    out = {}
    for i, rec in enumerate(records):
        key = rec.get("record_id") or f"index:{i}"
        out[key] = validate_record(rec)
    return out


def check_traceable(record: dict, manifest_path: Path | None = None) -> list[str]:
    """Conformance Level 2: every provenance hash must match an entry in MANIFEST.json."""
    from neuromps.fetch.manifest import Manifest

    path = manifest_path or REPO_ROOT / "data" / "raw" / "MANIFEST.json"
    known = {f.sha256: f.path for f in Manifest.load(path).file_index().values()}
    errors = []
    for src in record.get("provenance", {}).get("source_files", []):
        if src["sha256"] not in known:
            errors.append(f"provenance: {src['path']} sha256 not found in {path.name}")
        elif known[src["sha256"]] != src["path"]:
            errors.append(
                f"provenance: {src['path']} hash belongs to {known[src['sha256']]} in {path.name}"
            )
    return errors


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m neuromps.qc.schema [--traceable] FILE...``."""
    import argparse

    p = argparse.ArgumentParser(description="Validate NeuroMPS records against the schema.")
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument(
        "--traceable", action="store_true", help="also check hashes against MANIFEST.json"
    )
    args = p.parse_args(argv)
    failed = 0
    for path in args.files:
        record = json.loads(path.read_text())
        errors = validate_record(record) + (check_traceable(record) if args.traceable else [])
        status = "OK" if not errors else f"{len(errors)} error(s)"
        print(f"{path.name:<42} {status}")
        for e in errors:
            print(f"    {e}")
        failed += bool(errors)
    print(f"\n{len(args.files) - failed}/{len(args.files)} valid")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

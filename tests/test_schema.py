"""Tests for the NeuroMPS record schema and its validator."""

import copy
import json
from pathlib import Path

import pytest

from neuromps.qc.schema import EXAMPLES_DIR, load_schema, validate_record, validate_records


def _valid() -> dict:
    return json.loads((EXAMPLES_DIR / "kapucu_spike_times.json").read_text())


def test_schema_is_valid_json_schema() -> None:
    import jsonschema

    schema = load_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema["$id"].endswith("neuromps_schema.json")


def test_all_examples_validate() -> None:
    examples = sorted(EXAMPLES_DIR.glob("*.json"))
    assert len(examples) >= 4, "one example per data level and platform"
    for path in examples:
        assert validate_record(json.loads(path.read_text())) == [], path.name


def test_examples_cover_every_data_level() -> None:
    levels = {json.loads(p.read_text())["data_level"] for p in EXAMPLES_DIR.glob("*.json")}
    assert levels == {"spike_times", "per_well_features", "fitted_curve"}


def test_missing_required_field_is_reported() -> None:
    rec = _valid()
    del rec["license"]
    assert any("license" in e for e in validate_record(rec))


def test_unknown_field_is_rejected() -> None:
    rec = _valid()
    rec["firing_rate"] = 1.0  # typo for features.mfr_hz
    assert any("firing_rate" in e for e in validate_record(rec))


def test_species_enum_is_enforced() -> None:
    rec = _valid()
    rec["cell_source"]["species"] = "ferret"
    assert any("ferret" in e for e in validate_record(rec))


def test_negative_firing_rate_is_rejected() -> None:
    rec = _valid()
    rec["features"]["mfr_hz"] = -1.0
    assert validate_record(rec)


def test_percentage_above_100_is_rejected() -> None:
    rec = _valid()
    rec["features"]["pct_spikes_in_bursts"] = 120.0
    assert validate_record(rec)


def test_spike_times_level_requires_spike_detection_provenance() -> None:
    """Features we compute must say whose spike detection produced the input."""
    rec = _valid()
    del rec["feature_method"]["spike_detection"]
    assert any("spike_detection" in e for e in validate_record(rec))


def test_fitted_curve_level_requires_curve_block() -> None:
    rec = json.loads((EXAMPLES_DIR / "toxcast_fitted_curve.json").read_text())
    del rec["curve"]
    assert any("curve" in e for e in validate_record(rec))


def test_dose_requires_unit() -> None:
    rec = _valid()
    rec["compound"]["dose_value"] = 30.0
    rec["compound"].pop("dose_unit", None)
    assert any("dose_unit" in e for e in validate_record(rec))


def test_synchrony_value_requires_named_method() -> None:
    """STTC and Axion's synchrony index are different measures; the method is mandatory."""
    rec = _valid()
    rec["features"]["synchrony_index"] = 0.4
    rec["feature_method"].pop("synchrony_method", None)
    assert any("synchrony_method" in e for e in validate_record(rec))


def test_provenance_requires_source_file_hashes() -> None:
    rec = _valid()
    rec["provenance"]["source_files"] = []
    assert validate_record(rec)


def test_validate_records_keys_errors_by_record_id() -> None:
    good = _valid()
    bad = copy.deepcopy(good)
    bad["record_id"] = "bad-1"
    del bad["platform"]
    out = validate_records([good, bad])
    assert out[good["record_id"]] == []
    assert any("platform" in e for e in out["bad-1"])


def test_example_files_live_under_schema_dir() -> None:
    assert EXAMPLES_DIR.is_dir()
    assert EXAMPLES_DIR.parent.name == "schema"


def test_traceable_accepts_examples_against_real_manifest() -> None:
    """Conformance Level 2: example provenance hashes exist in data/raw/MANIFEST.json."""
    from neuromps.qc.schema import check_traceable

    manifest = Path(__file__).resolve().parents[1] / "data" / "raw" / "MANIFEST.json"
    if not manifest.exists():
        pytest.skip("raw manifest absent; run python -m neuromps.fetch.runner")
    for path in sorted(EXAMPLES_DIR.glob("*.json")):
        assert check_traceable(json.loads(path.read_text()), manifest) == [], path.name


def test_traceable_rejects_unknown_hash(tmp_path: Path) -> None:
    from neuromps.qc.schema import check_traceable

    manifest = tmp_path / "MANIFEST.json"
    manifest.write_text(json.dumps({"schema_version": 1, "sources": {}}))
    rec = _valid()
    assert check_traceable(rec, manifest)

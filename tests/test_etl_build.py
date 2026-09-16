"""Tests for ToxCast record building and the asset build pipeline."""

import json
from pathlib import Path

import pandas as pd
import pytest

from neuromps.etl.build import build_asset, flatten_records
from neuromps.etl.toxcast_mea import build_curve_records, select_mea
from neuromps.qc.schema import validate_record

MC56_ROW = {
    "aeid": 2456,
    "aenm": "CCTE_Shafer_MEA_acute_firing_rate_mean",
    "chid": 7,
    "spid": "TP0001",
    "chnm": "Fipronil",
    "casn": "120068-37-3",
    "hitc": 0.97,
    "top": 99.99,
    "ac50": 9.11,
    "coff": 70.97,
    "conc_min": 0.03,
    "conc_max": 40.0,
    "nconc": 7,
    "modl": "hill",
}


def _mc56(**overrides) -> pd.DataFrame:
    return pd.DataFrame([{**MC56_ROW, **overrides}])


def test_curve_records_validate(tmp_path: Path) -> None:
    recs = build_curve_records(select_mea(_mc56()), {"x.csv": "b" * 64}, source_file="x.csv")
    assert len(recs) == 1
    assert validate_record(recs[0]) == []


def test_curve_record_carries_normalized_direction() -> None:
    rec = build_curve_records(select_mea(_mc56()), {"x.csv": "b" * 64}, source_file="x.csv")[0]
    assert rec["data_level"] == "fitted_curve"
    assert rec["curve"]["top"] == 99.99
    assert rec["curve"]["activity_change"] == -99.99  # acute: +top means activity fell
    assert rec["compound"]["cas"] == "120068-37-3"


def test_curve_record_drops_an_unparseable_cas() -> None:
    rec = build_curve_records(select_mea(_mc56(casn="NOCAS")), {"x.csv": "b" * 64}, "x.csv")[0]
    assert rec["compound"]["cas"] is None
    assert validate_record(rec) == []


def test_control_rows_take_their_name_from_spid() -> None:
    """Control rows carry no chnm; DMSO is a vehicle, bicuculline a reference antagonist."""
    vehicle = build_curve_records(
        select_mea(_mc56(chnm=None, spid="DMSO", casn="NOCAS")), {"x.csv": "b" * 64}, "x.csv"
    )[0]
    assert vehicle["compound"]["name"] == "DMSO"
    assert vehicle["compound"]["role"] == "vehicle_control"
    bic = build_curve_records(
        select_mea(_mc56(chnm=None, spid="Bicuculline", casn="NOCAS")), {"x.csv": "b" * 64}, "x.csv"
    )[0]
    assert bic["compound"]["role"] == "positive_control"
    assert bic["compound"]["mechanism_class"] == "gabaa_antagonist"
    assert validate_record(bic) == []


def test_single_concentration_fits_are_flagged_not_dropped() -> None:
    rec = build_curve_records(select_mea(_mc56(nconc=1)), {"x.csv": "b" * 64}, "x.csv")[0]
    assert rec["curve"]["n_concentrations"] == 1
    assert rec["qc"]["usable"] is False
    assert "incomplete_dose_series" in rec["qc"]["flags"]
    assert validate_record(rec) == []


def test_record_id_is_slugified_for_awkward_spids() -> None:
    rec = build_curve_records(
        select_mea(_mc56(chnm=None, spid="DMSO/Ethanol", casn="NOCAS")),
        {"x.csv": "b" * 64},
        "x.csv",
    )[0]
    assert rec["record_id"] == "toxcast_mea:aeid2456:dmso-ethanol"
    assert validate_record(rec) == []


def test_flatten_records_gives_one_row_per_record_with_dotted_columns() -> None:
    recs = build_curve_records(select_mea(_mc56()), {"x.csv": "b" * 64}, "x.csv")
    df = flatten_records(recs)
    assert len(df) == 1
    assert df.loc[0, "compound.name"] == "Fipronil"
    assert df.loc[0, "cell_source.species"] == "rat"
    assert "provenance" not in df.columns  # nested provenance stays in the JSONL


def test_build_asset_writes_parquet_jsonl_and_provenance(tmp_path: Path) -> None:
    recs = build_curve_records(select_mea(_mc56()), {"x.csv": "b" * 64}, "x.csv")
    out = build_asset(recs, out_dir=tmp_path, check_hashes=False)
    assert (tmp_path / "neuromps.parquet").exists()
    assert (tmp_path / "neuromps_records.jsonl").exists()
    prov = json.loads((tmp_path / "provenance.json").read_text())
    assert prov["n_records"] == 1
    assert prov["counts"]["by_source"]["toxcast_mea"] == 1
    assert out["n_records"] == 1
    written = pd.read_parquet(tmp_path / "neuromps.parquet")
    assert len(written) == 1


def test_build_asset_refuses_invalid_records(tmp_path: Path) -> None:
    recs = build_curve_records(select_mea(_mc56()), {"x.csv": "b" * 64}, "x.csv")
    recs[0]["cell_source"]["species"] = "ferret"
    with pytest.raises(ValueError, match="failed schema validation"):
        build_asset(recs, out_dir=tmp_path, check_hashes=False)
    assert not (tmp_path / "neuromps.parquet").exists()


def test_build_asset_refuses_records_that_are_not_traceable(tmp_path: Path) -> None:
    """Conformance level 2 is enforced at build time, not just by the CLI."""
    recs = build_curve_records(select_mea(_mc56()), {"x.csv": "b" * 64}, "x.csv")
    manifest = tmp_path / "MANIFEST.json"
    manifest.write_text(json.dumps({"schema_version": 1, "sources": {}}))
    with pytest.raises(ValueError, match="not traceable"):
        build_asset(recs, out_dir=tmp_path, manifest_path=manifest)
    assert not (tmp_path / "neuromps.parquet").exists()

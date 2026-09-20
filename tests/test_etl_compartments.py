"""Tests for converting two/three-compartment ("Joint") devices into records."""

import zipfile
from pathlib import Path

import pytest

from neuromps.etl.screenneuropharm import (
    COMPARTMENT_SYSTEM,
    HETEROGENEOUS,
    build_compartment_records,
    discover_joint_devices,
)
from neuromps.qc.schema import validate_record

ROOT = "ScreenNeuroPharm-Bicuculline-6a1949e/Data"


def _ptrain(step: int, n: int = 600, total: int = 4_799_001) -> str:
    return f"{total},0\n" + "".join(f"{i * step},31.0\n" for i in range(1, n))


def _write_device(z: zipfile.ZipFile, region: str, array: str) -> None:
    base = f"{ROOT}/Br18/{region}/{array}/20210329_PeakDetectionMAT_PLP2ms_RP1ms_Phases_Joint_TXT"
    for cond in ("01_nbasal", "05_BIC30uM"):
        for colour, electrodes in (
            ("Red", ("A04", "A05")),
            ("Green", ("A08", "A09")),
            ("Gray", ("A06", "A07")),
        ):
            for el in electrodes:
                folder = f"{base}/ptrain_20210329_02_{cond}_{colour}"
                z.writestr(f"{folder}/ptrain_20210329_02_{cond}_Joint_{el}.txt", _ptrain(700))


@pytest.fixture()
def archive(tmp_path: Path) -> Path:
    path = tmp_path / "bic.zip"
    with zipfile.ZipFile(path, "w") as z:
        _write_device(z, "CxCx", "24544")
        _write_device(z, "HpHp", "35882")
        _write_device(z, "CxHp", "39173")  # heterogeneous: colour->region unknown
    return path


def test_compartment_system_map_covers_homogeneous_devices_only() -> None:
    assert COMPARTMENT_SYSTEM["CxCx"] == "primary_cortical"
    assert COMPARTMENT_SYSTEM["HpHp"] == "primary_hippocampal"
    assert "CxHp" not in COMPARTMENT_SYSTEM


def test_heterogeneous_devices_are_kept_but_labelled_unknown(archive: Path) -> None:
    """CxHp is the asymmetric case; keep it, but never guess which colour is which region."""
    with zipfile.ZipFile(archive) as z:
        devices = discover_joint_devices(z)
    assert {d.region for d in devices} == {"CxCx", "HpHp", "CxHp"}
    records = [r for r in build_compartment_records(archive, {}) if "cxhp" in r["record_id"]]
    assert records
    for rec in records:
        assert rec["cell_source"]["system"] == "unknown"
        assert HETEROGENEOUS["CxHp"][:30] in rec["cell_source"]["note"]
        assert validate_record(rec) == []


def test_one_record_per_compartment_per_condition(archive: Path) -> None:
    records = build_compartment_records(archive, hashes={})
    # 3 devices x 2 conditions x 3 compartments
    assert len(records) == 18
    ids = {(r["device"]["device_id"], r["device"]["compartment_id"]) for r in records}
    assert len(ids) == 9


def test_records_validate_and_carry_device_geometry(archive: Path) -> None:
    for rec in build_compartment_records(archive, hashes={"bic.zip": "f" * 64}):
        assert validate_record(rec) == [], rec["record_id"]
        assert rec["device"]["layout"] == "three_chamber_microchannel"
        assert rec["device"]["microchannel_count"] == 28
        assert rec["unit_id"].endswith(rec["device"]["compartment_id"])


def test_interface_strip_is_labelled_as_such(archive: Path) -> None:
    records = {(r["device"]["compartment_id"]): r for r in build_compartment_records(archive, {})}
    assert records["gray"]["device"]["compartment_role"] == "interface"
    assert records["red"]["device"]["compartment_role"] == "chamber"
    assert records["green"]["device"]["compartment_role"] == "chamber"


def test_coupling_is_recorded_towards_the_other_compartments(archive: Path) -> None:
    rec = next(
        r for r in build_compartment_records(archive, {}) if r["device"]["compartment_id"] == "red"
    )
    targets = {c["to_compartment"] for c in rec["coupling"]}
    assert targets == {"green", "gray"}
    assert "red" not in targets
    assert all("cross_correlation" in c["method"] for c in rec["coupling"])

"""Tests for the ScreenNeuroPharm bicuculline ETL (peak trains inside a zip)."""

import zipfile
from pathlib import Path

import numpy as np
import pytest

from neuromps.etl.screenneuropharm import (
    SAMPLING_HZ,
    build_records,
    discover_recordings,
    parse_condition,
    parse_ptrain,
)
from neuromps.qc.schema import validate_record

ROOT = "ScreenNeuroPharm-Bicuculline-6a1949e/Data"
SINGLE = f"{ROOT}/Br11/Cx/18331/20092022_PeakDetectionMAT_PLP2ms_RP1ms_Cut_TXT"
JOINT = f"{ROOT}/Br18/CxHp/39173/20210329_PeakDetectionMAT_PLP2ms_RP1ms_Phases_Joint_TXT"


def _ptrain(spike_samples: list[int], total: int = 4_799_001) -> str:
    return f"{total},0\n" + "".join(f"{s},30.5\n" for s in spike_samples)


@pytest.fixture()
def archive(tmp_path: Path) -> Path:
    path = tmp_path / "bic.zip"
    with zipfile.ZipFile(path, "w") as z:
        for condition in ("01_nbasal", "02_BIC100nM", "03_BIC10uM"):
            for electrode in range(12):
                folder = f"{SINGLE}/ptrain_20092022_01_{condition}"
                name = f"{folder}/ptrain_20092022_01_{condition}_{electrode:03d}.txt"
                z.writestr(name, _ptrain(list(range(1000, 400_000, 2_000))))
        for colour in ("Gray", "Green", "Red"):
            folder = f"{JOINT}/ptrain_20210329_02_01_nbasal_{colour}"
            name = f"{folder}/ptrain_20210329_02_01_nbasal_Joint_A06.txt"
            z.writestr(name, _ptrain([1000, 2000]))
    return path


def test_parse_ptrain_converts_samples_to_seconds() -> None:
    times, duration = parse_ptrain(_ptrain([10_000, 20_000], total=4_799_001))
    assert np.allclose(times, [1.0, 2.0])
    assert duration == pytest.approx(4_799_001 / SAMPLING_HZ)


def test_parse_ptrain_handles_an_electrode_with_no_spikes() -> None:
    times, duration = parse_ptrain(_ptrain([]))
    assert times.size == 0
    assert duration > 0


def test_parse_condition_reads_dose_and_unit() -> None:
    assert parse_condition("ptrain_20092022_01_02_BIC100nM") == {
        "name": "bicuculline",
        "role": "test",
        "mechanism_class": "gabaa_antagonist",
        "dose_value": 100.0,
        "dose_unit": "nM",
    }
    assert parse_condition("ptrain_20210329_06_04_1uM")["dose_value"] == 1.0
    assert parse_condition("ptrain_20210329_06_04_1uM")["dose_unit"] == "uM"


def test_parse_condition_treats_basal_as_an_untreated_control() -> None:
    cond = parse_condition("ptrain_20092022_01_01_nbasal")
    assert cond["role"] == "untreated_control"
    assert cond["dose_value"] is None


def test_discover_skips_two_compartment_devices(archive: Path) -> None:
    """Two-compartment (CxHp/CxCx/HpHp) devices need a schema extension; not yet in the asset."""
    with zipfile.ZipFile(archive) as z:
        found = discover_recordings(z)
    assert len(found) == 3
    assert all(rec.region in ("Cx", "Hp") for rec in found)
    assert not any("Joint" in m for rec in found for m in rec.members)


def test_records_validate_and_carry_species_and_platform(archive: Path) -> None:
    records = build_records(archive, hashes={"bic.zip": "d" * 64})
    assert len(records) == 3
    for rec in records:
        assert validate_record(rec) == [], rec["record_id"]
        assert rec["cell_source"]["species"] == "rat"
        assert rec["cell_source"]["system"] == "primary_cortical"
        assert rec["platform"]["vendor"] == "Multi Channel Systems"
        assert rec["recording"]["duration_s"] == pytest.approx(479.9, abs=0.1)


def test_basal_and_dose_records_differ_in_phase_and_compound(archive: Path) -> None:
    records = {r["compound"]["dose_value"]: r for r in build_records(archive, hashes={})}
    assert records[None]["recording"]["phase"] == "baseline"
    assert records[100.0]["recording"]["phase"] == "exposure"
    assert records[10.0]["compound"]["dose_unit"] == "uM"
    assert records[100.0]["compound"]["dose_unit"] == "nM"


def test_feature_cache_is_reused(archive: Path, tmp_path: Path) -> None:
    cache = tmp_path / "snp_features.parquet"
    build_records(archive, hashes={}, cache_path=cache)
    assert cache.exists()
    import pandas as pd

    cached = pd.read_parquet(cache)
    cached.loc[:, "mfr_hz"] = 99.0
    cached.to_parquet(cache, index=False)
    again = build_records(archive, hashes={}, cache_path=cache)
    assert all(r["features"]["mfr_hz"] == 99.0 for r in again)

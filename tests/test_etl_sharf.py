"""Tests for the Sharf 2022 organoid HD-MEA ETL (Kilosort2 .mat inside a zip)."""

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio

from neuromps.etl.sharf import (
    FS,
    build_records,
    parse_condition,
    spikes_by_electrode,
    wanted_members,
)
from neuromps.qc.schema import validate_record


def _mat(units: list[tuple[int, list[int]]]) -> bytes:
    """MATLAB file with the fields the real archive uses."""
    buf = io.BytesIO()
    sio.savemat(
        buf,
        {
            "fs": float(FS),
            "locations": np.zeros((4, 2)),
            "units": np.array(
                [
                    {"spike_train": np.array(train), "electrode": el, "unit_id": i}
                    for i, (el, train) in enumerate(units)
                ]
            ),
        },
    )
    return buf.getvalue()


@pytest.fixture()
def archive(tmp_path: Path) -> Path:
    path = tmp_path / "kilosort2.zip"
    train = list(range(0, 180 * FS, FS // 4))  # 4 Hz for 180 s
    payload = _mat([(11, train), (11, train), (12, train), (13, train[:2])])
    with zipfile.ZipFile(path, "w") as z:
        for name in ("control_2953", "diazepam3uM_2953", "diazepam50uM_2953"):
            z.writestr(f"kilosort2/drug/kilosort2/{name}.mat", payload)
        z.writestr("kilosort2/7month/kilosort2/2953.mat", payload)
        z.writestr("kilosort2/development/kilosort2/2953_5month.mat", payload)
        z.writestr("kilosort2/interval/kilosort2/4hr_1200_2950.mat", payload)
        z.writestr("kilosort2/neuropixels/kilosort2/np_2953.mat", payload)
    return path


def test_wanted_members_skips_neuropixels_and_interval(archive: Path) -> None:
    with zipfile.ZipFile(archive) as z:
        members = wanted_members(z)
    groups = {m.split("/")[1] for m in members}
    assert groups == {"drug", "7month", "development"}


def test_spikes_grouped_by_electrode_not_by_sorted_unit() -> None:
    mat = sio.loadmat(
        io.BytesIO(_mat([(11, [0, FS]), (11, [2 * FS]), (12, [3 * FS])])),
        squeeze_me=True,
        struct_as_record=False,
    )
    spikes, duration = spikes_by_electrode(mat)
    assert set(spikes) == {"11", "12"}
    assert np.allclose(sorted(spikes["11"]), [0.0, 1.0, 2.0])  # two units merged
    assert duration == pytest.approx(3.0)


def test_coincident_spikes_on_one_electrode_collapse() -> None:
    """A sorter can split one neuron into two units; the electrode saw one spike."""
    mat = sio.loadmat(
        io.BytesIO(_mat([(11, [0, FS]), (11, [10, FS + 5])])),  # within 1 ms of the first unit
        squeeze_me=True,
        struct_as_record=False,
    )
    spikes, _ = spikes_by_electrode(mat)
    assert spikes["11"].size == 2


def test_parse_condition_reads_dose_and_organoid() -> None:
    assert parse_condition("diazepam50uM_2953", "drug") == (
        "2953",
        {
            "name": "diazepam",
            "role": "test",
            "mechanism_class": "gabaa_agonist_or_pam",
            "dose_value": 50.0,
            "dose_unit": "uM",
        },
        None,
    )


def test_parse_condition_marks_controls_and_reads_age_in_months() -> None:
    organoid, compound, age = parse_condition("control_2953", "drug")
    assert (organoid, compound["role"], age) == ("2953", "untreated_control", None)
    organoid, _, age = parse_condition("2953_5month", "development")
    assert (organoid, age) == ("2953", 150.0)  # 5 months x 30 days


def test_records_validate_and_describe_the_platform(archive: Path) -> None:
    records = build_records(archive, hashes={"kilosort2.zip": "e" * 64})
    assert len(records) == 5
    for rec in records:
        assert validate_record(rec) == [], rec["record_id"]
        assert rec["cell_source"]["species"] == "human"
        assert rec["cell_source"]["system"] == "brain_organoid_slice"
        assert rec["platform"]["vendor"] == "MaxWell Biosystems"
        assert "Kilosort2" in rec["feature_method"]["spike_detection"]


def test_records_use_the_shared_extractor_and_count_active_electrodes(archive: Path) -> None:
    rec = next(r for r in build_records(archive, hashes={}) if "diazepam3um" in r["record_id"])
    assert rec["feature_method"]["computed_by"] == "neuromps.features.mea"
    assert rec["features"]["active_electrodes"] == 2  # electrode 13 has 2 spikes: inactive
    # electrode 11 carries two identical units, collapsed to one 4 Hz train
    assert rec["features"]["mfr_hz"] == pytest.approx(4.0, rel=0.1)


def test_cache_is_reused(archive: Path, tmp_path: Path) -> None:
    cache = tmp_path / "sharf.parquet"
    build_records(archive, hashes={}, cache_path=cache)
    assert cache.exists()
    import pandas as pd

    table = pd.read_parquet(cache)
    table.loc[:, "mfr_hz"] = 42.0
    table.to_parquet(cache, index=False)
    assert all(
        r["features"]["mfr_hz"] == 42.0 for r in build_records(archive, {}, cache_path=cache)
    )

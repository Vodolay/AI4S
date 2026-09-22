"""Reading Trujillo organoid sessions, and turning them into records.

The extractor is checked against a synthetic file built the way MATLAB v7.3
writes one - object references into a `#refs#`-style group, a timebase that
implies the sample rate, and empty cells for silent electrodes - so the parsing
is tested without the network or a gigabyte of LFP.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from neuromps.etl import trujillo
from neuromps.fetch.trujillo2019 import _is_empty, session_spikes

SAMPLE_RATE = 12500.0
DURATION = 4.0


@pytest.fixture
def session_file(tmp_path: Path) -> Path:
    """A miniature session: 3 electrodes x 2 wells, one of them silent."""
    path = tmp_path / "LFP_Sp_990101.mat"
    n_samples = int(DURATION * SAMPLE_RATE) + 1
    with h5py.File(path, "w") as f:
        f.create_dataset("t_s", data=np.linspace(0, DURATION, n_samples)[None, :])
        refs = f.create_group("cells")
        spikes = np.empty((3, 2), dtype=h5py.ref_dtype)
        for electrode in range(3):
            for well in range(2):
                name = f"c{electrode}_{well}"
                if well == 1 and electrode == 2:  # a silent electrode
                    cell = refs.create_dataset(name, data=np.zeros((2, 1)))
                    cell.attrs["MATLAB_empty"] = 1
                else:
                    samples = np.arange(1, 5) * (electrode + 1) * 100.0
                    cell = refs.create_dataset(name, data=samples[:, None])
                spikes[electrode, well] = cell.ref
        f.create_dataset("spikes", data=spikes)
    return path


def test_the_sample_rate_is_recovered_from_the_timebase(session_file: Path) -> None:
    """Spike times are sample indices, so the rate must not be assumed."""
    with h5py.File(session_file, "r") as f:
        _, sample_rate_hz, duration_s = session_spikes(f)
    assert sample_rate_hz == pytest.approx(SAMPLE_RATE)
    assert duration_s == pytest.approx(DURATION)


def test_spike_indices_become_seconds(session_file: Path) -> None:
    with h5py.File(session_file, "r") as f:
        frame, _, _ = session_spikes(f)
    first = frame[(frame.well == 0) & (frame.electrode == 0)].time_s.to_numpy()
    assert first == pytest.approx(np.array([100, 200, 300, 400]) / SAMPLE_RATE, rel=1e-5)


def test_a_silent_electrode_contributes_no_rows(session_file: Path) -> None:
    with h5py.File(session_file, "r") as f:
        frame, _, _ = session_spikes(f)
    assert frame[(frame.well == 1) & (frame.electrode == 2)].empty
    assert len(frame) == 5 * 4, "five live cells of four spikes each"


def test_an_empty_cell_is_recognised_by_attribute_and_by_shape(tmp_path: Path) -> None:
    with h5py.File(tmp_path / "e.h5", "w") as f:
        tagged = f.create_dataset("tagged", data=np.zeros((2, 1)))
        tagged.attrs["MATLAB_empty"] = 1
        assert _is_empty(tagged)
        assert _is_empty(f.create_dataset("bare", data=np.zeros((2, 1))))
        assert not _is_empty(f.create_dataset("real", data=np.array([[5.0], [7.0]])))


@pytest.mark.parametrize(
    ("name", "expected_session", "expected_treated"),
    [
        ("LFP_Sp_160810.spikes.parquet", "160810", False),
        ("LFP_Sp_161217_Drugs.spikes.parquet", "161217_drugs", True),
    ],
)
def test_session_ids_are_lowercase_and_flag_the_drug_sessions(
    name: str, expected_session: str, expected_treated: bool
) -> None:
    session, treated = trujillo.session_id(Path(name))
    assert (session, treated) == (expected_session, expected_treated)


def test_an_unexpected_file_name_is_refused() -> None:
    with pytest.raises(ValueError, match="unexpected"):
        trujillo.session_id(Path("something_else.parquet"))


def test_a_drug_session_is_flagged_but_never_labelled() -> None:
    """The deposit has no well-to-compound mapping, so a mechanism would be invented."""
    treated = trujillo._compound(True)
    assert treated["role"] == "test"
    assert treated["mechanism_class"] is None
    assert treated["dose_value"] is None
    untreated = trujillo._compound(False)
    assert untreated["role"] == "untreated_control"
    assert untreated["mechanism_class"] is None


def test_a_session_without_duration_metadata_is_refused(tmp_path: Path) -> None:
    """A duration inferred from the last spike would bias every rate upwards."""
    import pandas as pd

    path = tmp_path / "LFP_Sp_990101.spikes.parquet"
    pd.DataFrame({"well": [0], "electrode": [0], "time_s": [1.0]}).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="duration"):
        trujillo.read_session(path)


def test_records_carry_the_organoid_as_a_stable_unit_id(session_file: Path, tmp_path: Path) -> None:
    """The same well is the same organoid week after week; the bootstrap groups on it."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    with h5py.File(session_file, "r") as f:
        frame, rate, duration = session_spikes(f)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    table = table.replace_schema_metadata(
        {b"duration_s": f"{duration}".encode(), b"sample_rate_hz": f"{rate}".encode()}
    )
    out = tmp_path / "spikes"
    out.mkdir()
    pq.write_table(table, out / "LFP_Sp_990101.spikes.parquet")
    # Keyed by file name, as `etl.build.manifest_hashes` produces.
    records = trujillo.build_records(out, hashes={"LFP_Sp_990101.spikes.parquet": "x"})
    assert {r["unit_id"] for r in records} == {"well00", "well01"}
    assert all(r["cell_source"]["species"] == "human" for r in records)
    assert all(r["recording"]["duration_s"] == pytest.approx(DURATION) for r in records)
    assert all(r["culture_age_days"] is None for r in records)


def test_an_unlabelled_exposure_never_becomes_a_control_pair() -> None:
    """A treated recording must not teach the classifier what untreated looks like."""
    import pandas as pd

    from neuromps.models.dataset import build_pairs, is_unlabelled_exposure

    common = {
        "data_level": "spike_times",
        "qc.usable": True,
        "source": "trujillo2019",
        "cell_source.species": "human",
        "cell_source.system": "hpsc_derived_cortical",
        "unit_id": "well04",
        "plate_id": "161217",
        "device.compartment_id": None,
        "features.mfr_hz": 1.0,
        "features.burst_rate_per_min": 1.0,
        "features.synchrony_index": 0.5,
        "features.isi_cv": 1.0,
        "features.pct_spikes_in_bursts": 10.0,
        "features.network_burst_rate_per_min": 1.0,
        "features.burst_duration_s": 0.1,
        "features.spikes_per_burst": 5.0,
        "features.active_electrodes": 20,
    }
    baseline = {
        **common,
        "record_id": "trujillo2019:161217:w04",
        "recording.phase": "baseline",
        "compound.role": "untreated_control",
        "compound.mechanism_class": None,
        "compound.name": "none",
    }
    treated = {
        **common,
        "record_id": "trujillo2019:161217_drugs:w04",
        "recording.phase": "exposure",
        "compound.role": "test",
        "compound.mechanism_class": None,
        "compound.name": "unspecified_in_deposit",
    }
    assert is_unlabelled_exposure(pd.Series(treated))
    assert not is_unlabelled_exposure(pd.Series(baseline))
    pairs = build_pairs(pd.DataFrame([baseline, treated]))
    assert pairs.empty, "an unidentified compound must not be paired as a control"


def test_extraction_is_checked_against_the_deposits_own_spike_counts(tmp_path) -> None:
    """A dropped spike must fail the fetch, not reach the asset quietly."""
    import h5py
    import numpy as np

    from neuromps.fetch.trujillo2019 import session_spikes

    path = tmp_path / "wrong.mat"
    with h5py.File(path, "w") as f:
        f.create_dataset("t_s", data=np.linspace(0, 1.0, 12501)[None, :])
        refs = f.create_group("cells")
        spikes = np.empty((2, 1), dtype=h5py.ref_dtype)
        for electrode in range(2):
            cell = refs.create_dataset(f"c{electrode}", data=np.arange(1, 4, dtype=float)[:, None])
            spikes[electrode, 0] = cell.ref
        f.create_dataset("spikes", data=spikes)
        # The deposit claims four spikes on the first electrode; we will read three.
        f.create_dataset("spike_cnt", data=np.array([[4.0], [3.0]]))
    with h5py.File(path, "r") as f, pytest.raises(ValueError, match="disagree with the deposit"):
        session_spikes(f)


def test_a_session_that_agrees_with_its_counts_passes(session_file: Path) -> None:
    import h5py
    import numpy as np

    from neuromps.fetch.trujillo2019 import session_spikes

    with h5py.File(session_file, "r+") as f:
        counts = np.zeros((3, 2))
        for electrode in range(3):
            for well in range(2):
                counts[electrode, well] = 0 if (well == 1 and electrode == 2) else 4
        f.create_dataset("spike_cnt", data=counts)
    with h5py.File(session_file, "r") as f:
        frame, _, _ = session_spikes(f)
    assert len(frame) == 20


def test_a_session_absent_from_this_tiers_manifest_is_skipped(session_file: Path, tmp_path) -> None:
    """A --full extraction seen from a --quick run must not become a record.

    The file is on disk and this tier's manifest says nothing about it, so a
    record built from it fails the conformance check - which is how this was
    found, from a message about hashes rather than about tiers.
    """
    import h5py
    import pyarrow as pa
    import pyarrow.parquet as pq

    from neuromps.fetch.trujillo2019 import session_spikes

    with h5py.File(session_file, "r") as f:
        frame, rate, duration = session_spikes(f)
    table = pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(
        {b"duration_s": f"{duration}".encode(), b"sample_rate_hz": f"{rate}".encode()}
    )
    out = tmp_path / "spikes"
    out.mkdir()
    pq.write_table(table, out / "LFP_Sp_990101.spikes.parquet")
    assert trujillo.build_records(out, hashes={"some_other_file.parquet": "x"}) == []
    assert trujillo.build_records(out, hashes={"LFP_Sp_990101.spikes.parquet": "x"})

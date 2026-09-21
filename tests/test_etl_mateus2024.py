"""Tests for recovering microchannel propagation from the Mateus deposit.

The deposit publishes the measurement but not the electrode map, so the geometry
is recovered from the spike times themselves. That makes the detector the load-
bearing piece of this source, and it is tested against trains whose direction is
known by construction before it is pointed at real recordings.
"""

import numpy as np
import pytest

from neuromps.etl.mateus2024 import (
    MAX_STEP_S,
    channel_runs,
    chip_summary,
    grid_position,
    ordered_events,
    propagation,
)


def _chain(n_events: int, step_s: float, start: float = 1.0, jitter: float = 0.0, seed: int = 0):
    """Five trains carrying `n_events` propagations, each step `step_s` apart."""
    rng = np.random.default_rng(seed)
    times = start + np.arange(n_events) * 0.5
    trains = []
    for position in range(5):
        offset = position * step_s + (rng.normal(0, jitter, n_events) if jitter else 0)
        trains.append(np.sort(times + offset))
    return trains


def test_a_known_forward_chain_is_counted_forward() -> None:
    trains = _chain(20, step_s=0.0004)  # 0.4 ms per step, inside the 1 ms criterion
    forward, backward = propagation(trains)
    assert forward == 20
    assert backward == 0


def test_reversing_the_electrodes_reverses_the_direction() -> None:
    trains = _chain(20, step_s=0.0004)
    forward, backward = propagation(list(reversed(trains)))
    assert backward == 20
    assert forward == 0


def test_steps_slower_than_the_criterion_are_not_events() -> None:
    """The deposit requires <=1 ms per step; 2 ms is a different phenomenon."""
    forward, backward = propagation(_chain(20, step_s=0.002))
    assert forward == 0 and backward == 0


def test_the_event_must_appear_on_every_electrode() -> None:
    trains = _chain(20, step_s=0.0004)
    trains[2] = np.array([])  # one dead electrode in the middle of the channel
    assert propagation(trains) == (0, 0)


def test_independent_trains_give_no_systematic_direction() -> None:
    rng = np.random.default_rng(1)
    trains = [np.sort(rng.uniform(0, 600, 3000)) for _ in range(5)]
    forward, backward = propagation(trains)
    total = forward + backward
    assert total == 0 or abs(forward - backward) / total < 0.5


def test_jittered_but_real_propagation_still_counts() -> None:
    forward, backward = propagation(_chain(40, step_s=0.0004, jitter=0.00005, seed=3))
    assert forward > backward
    assert forward >= 30


def test_grid_positions_parse_the_mcs_labels() -> None:
    assert grid_position("A1") == (0, 1)
    assert grid_position("R16") == (15, 16)
    assert grid_position("J7") == (8, 7)  # there is no column I on an MCS array
    assert grid_position("nonsense") is None


def test_channel_runs_are_five_electrodes_in_one_column() -> None:
    runs = channel_runs(["F6", "F7", "F8", "F9", "F10", "F11"])
    assert ["F6", "F7", "F8", "F9", "F10"] in runs
    assert ["F7", "F8", "F9", "F10", "F11"] in runs
    assert all(len(run) == 5 for run in runs)


def test_a_column_with_a_gap_yields_no_run() -> None:
    assert channel_runs(["F6", "F7", "F9", "F10", "F11"]) == [["F7", "F8", "F9", "F10", "F11"]][0:0]


def test_chip_summary_reports_asymmetry_and_what_it_rests_on() -> None:
    spikes = {}
    for position, label in enumerate(["F6", "F7", "F8", "F9", "F10"]):
        spikes[label] = _chain(30, step_s=0.0004)[position]
    spikes["A1"] = np.sort(np.random.default_rng(0).uniform(0, 600, 200))
    out = chip_summary(spikes, duration_s=600.0)
    assert out["n_channels_detected"] == 1
    assert out["forward_events"] == 30
    assert out["backward_events"] == 0
    assert out["asymmetry"] == pytest.approx(1.0)


def test_a_chip_with_no_propagation_reports_no_asymmetry() -> None:
    rng = np.random.default_rng(2)
    spikes = {f"F{row}": np.sort(rng.uniform(0, 600, 500)) for row in range(1, 17)}
    out = chip_summary(spikes, duration_s=600.0)
    assert out["asymmetry"] is None
    assert out["n_channels_detected"] == 0


def test_the_step_limit_matches_the_published_criterion() -> None:
    assert pytest.approx(0.001) == MAX_STEP_S


def test_ordered_events_does_not_double_count_one_spike() -> None:
    """One source spike can start at most one chain, however dense the next train."""
    dense = np.linspace(1.0002, 1.0008, 25)
    trains = [np.array([1.0]), dense, dense + 0.0004, dense + 0.0008, dense + 0.0012]
    assert ordered_events(trains) <= 1


def test_overlapping_windows_collapse_to_one_channel() -> None:
    """Consecutive windows share four electrodes; one channel must count once."""
    from neuromps.etl.mateus2024 import _drop_overlaps

    strong = {"electrodes": ["F6", "F7", "F8", "F9", "F10"], "forward": 90, "backward": 5}
    shifted = {"electrodes": ["F7", "F8", "F9", "F10", "F11"], "forward": 70, "backward": 4}
    kept = _drop_overlaps([shifted, strong])
    assert kept == [strong]


def test_two_separate_channels_in_one_column_are_both_kept() -> None:
    from neuromps.etl.mateus2024 import _drop_overlaps

    lower = {"electrodes": ["F1", "F2", "F3", "F4", "F5"], "forward": 40, "backward": 2}
    upper = {"electrodes": ["F10", "F11", "F12", "F13", "F14"], "forward": 60, "backward": 3}
    kept = _drop_overlaps([lower, upper])
    assert len(kept) == 2


def _write_recording(path, entities, duration_us=600_100_000):
    """A minimal MCS-shaped file, for testing the reader without the deposit."""
    import h5py

    with h5py.File(path, "w") as f:
        recording = f.create_group("Data/Recording_0")
        recording.attrs["Duration"] = duration_us
        stream = recording.create_group("TimeStampStream/Stream_0")
        dtype = np.dtype(
            [
                ("TimeStampEntityID", "<i4"),
                ("GroupID", "<i4"),
                ("Label", h5py.special_dtype(vlen=str)),
                ("Unit", h5py.special_dtype(vlen=str)),
                ("Exponent", "<i4"),
                ("SourceChannelIDs", h5py.special_dtype(vlen=str)),
                ("SourceChannelLabels", h5py.special_dtype(vlen=str)),
            ]
        )
        info = np.array(
            [(i, 0, label, "s", -6, str(i), label) for i, (label, _) in enumerate(entities)],
            dtype=dtype,
        )
        stream.create_dataset("InfoTimeStamp", data=info)
        for i, (_, times_us) in enumerate(entities):
            stream.create_dataset(f"TimeStampEntity_{i}", data=np.array([times_us], dtype=np.int64))


def test_reading_converts_microseconds_to_seconds(tmp_path) -> None:
    from neuromps.etl.mateus2024 import read_recording

    path = tmp_path / "r.h5"
    _write_recording(path, [("F7", [1_000_000, 2_500_000]), ("F8", [1_000_400])])
    spikes, duration_s, unreadable = read_recording(path)
    assert duration_s == pytest.approx(600.1)
    assert np.allclose(spikes["F7"], [1.0, 2.5])
    assert np.allclose(spikes["F8"], [1.0004])
    assert unreadable == []


def test_an_unreadable_channel_is_reported_not_raised(tmp_path, monkeypatch) -> None:
    """Two channels in the deposit fail to decompress; 250 good ones must survive."""
    import h5py

    from neuromps.etl import mateus2024

    path = tmp_path / "r.h5"
    _write_recording(path, [("F7", [1_000_000]), ("F8", [1_000_400])])
    original = h5py.Dataset.__getitem__

    def flaky(self, item):
        if self.name.endswith("TimeStampEntity_1"):
            raise OSError("filter returned failure during read")
        return original(self, item)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", flaky)
    spikes, _, unreadable = mateus2024.read_recording(path)
    assert set(spikes) == {"F7"}
    assert unreadable == ["F8"]

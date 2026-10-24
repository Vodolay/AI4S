"""Stage 4a: conditions, the movement disruption index, onsets and timing, on synthetic inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from neuromps.fly import embodied_stage4a as s4a


def test_conditions_are_the_parents_with_their_knobs() -> None:
    conds = s4a.conditions()
    assert len(conds) == 22
    row = conds.set_index("condition").loc["ach_block_0.4"]
    assert row["knob"] == {"ach_scale": 0.4} and abs(row["dose"] - 0.6) < 1e-9
    assert conds.set_index("condition").loc["vehicle", "knob"] == {}


def test_disruption_is_zero_at_vehicle_and_one_when_a_readout_halves_everywhere() -> None:
    medians = pd.DataFrame({"a": [0.39, 0.19], "b": [0.99, 0.49]}, index=["vehicle", "half"])
    got = s4a.disruption(medians, eps=0.01)
    assert got["vehicle"] == 0.0 and abs(got["half"] - 1.0) < 1e-9


def test_onset_is_the_lowest_dose_reaching_the_line() -> None:
    table = pd.DataFrame(
        {
            "family": ["vehicle", "f", "f", "f", "g"],
            "dose": [0, 0.6, 0.2, 0.4, 0.5],
            "x": [0, 0.7, 0.1, 0.6, 0.2],
        }
    )
    assert s4a.onset(table, "x", 0.5) == {"f": 0.4, "g": None}


def test_flagged_in_time_compares_detection_with_movement_onset() -> None:
    table = pd.DataFrame(
        {"family": ["f", "f", "g"], "condition": ["f1", "f2", "g1"], "dose": [0.2, 0.4, 0.3]}
    )
    delta = {"f1": 0.1, "f2": 0.3, "g1": 0.9}
    got = s4a.flagged_in_time(table, delta, 0.2, {"f": 0.4, "g": 0.2, "h": None})
    assert got == {"f": True, "g": False}
    assert np.isclose(s4a.EPS_MM, 0.01)

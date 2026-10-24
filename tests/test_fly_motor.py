"""Motor-level readouts: pools, disruption and onset comparisons, on hand-made numbers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neuromps.fly import motor


def test_pools_take_only_motor_neurons_of_a_subclass() -> None:
    superclass = np.array(["vnc_motor", "vnc_sensory", "cb_motor", "vnc_motor"], dtype=object)
    subclass = np.array(["fl", "fl", "pm", "hl"], dtype=object)
    pools = motor.pools(superclass, subclass)
    assert pools["front_leg"].tolist() == [0]
    assert pools["proboscis"].tolist() == [2]
    assert pools["all_motor"].tolist() == [0, 2, 3]


def test_motor_disruption_is_the_mean_absolute_log_change_of_the_four_readouts() -> None:
    vehicle = dict.fromkeys(motor.READOUTS, 3.0)
    treated = {**vehicle, "proboscis_sugar": 15.0}
    assert motor.disruption(treated, vehicle) == pytest.approx(0.5)


def _table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "condition": ["vehicle", "f_0.8", "f_0.6", "g_0.8", "g_0.6"],
            "family": ["vehicle", "f", "f", "g", "g"],
            "dose": [0.0, 0.2, 0.4, 0.2, 0.4],
            "motor_disruption": [0.0, 0.6, 0.9, 0.1, 0.7],
            "disruption": [0.0, 0.2, 0.8, 0.6, 0.9],
            "abs_delta_mfr": [0.0, 0.1, 0.7, 0.6, 0.9],
        }
    )


def test_first_dose_reads_any_column() -> None:
    got = motor.first_doses(_table(), "motor_disruption", 0.5)
    assert got == {"f": 0.2, "g": 0.4}


def test_onset_comparison_counts_families_scored_at_both_levels() -> None:
    table = _table()
    got = motor.compare_onsets(table, "motor_disruption", "disruption", 0.5, 0.5, strict=False)
    assert got == {"meeting": 1, "scored": 2}  # f: motor 0.2 <= 0.4; g: motor 0.4 > 0.2


def test_the_chip_counts_as_late_when_it_never_detects() -> None:
    table = _table().assign(abs_delta_mfr=[0.0, 0.1, 0.2, 0.6, 0.9])
    got = motor.chip_late(table, 0.5, 0.5)
    assert got == {"meeting": 1, "scored": 2}  # f: chip never detects (late); g: chip at 0.2 < 0.4

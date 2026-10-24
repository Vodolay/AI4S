"""Scoring rules of the output-tract test, on hand-made numbers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from neuromps.fly import electrodes, tract


def _parent() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "condition": ["vehicle", "f_0.8", "f_0.6"],
            "family": ["vehicle", "f", "f"],
            "dose": [0.0, 0.2, 0.4],
            "disruption": [0.0, 0.7, 1.2],
            "delta.mfr_hz": [0.0, 0.1, 0.3],
        }
    )


def test_onset_is_the_first_dose_that_impairs_behaviour() -> None:
    assert tract.onset(_parent(), 0.5) == {"f": (0.2, "f_0.8")}


def test_a_family_is_flagged_in_time_only_if_the_threshold_is_reached_by_onset() -> None:
    first = tract.onset(_parent(), 0.5)
    assert tract.flagged_in_time(_parent(), {"f_0.8": 0.2, "f_0.6": 0.4}, first) == {"f": True}
    assert tract.flagged_in_time(_parent(), {"f_0.8": 0.1, "f_0.6": 0.4}, first) == {"f": False}


def test_subsets_and_tract_arrays_are_single_units_without_repeats() -> None:
    array = electrodes.VirtualArray(
        {f"A_{k}": np.array([k]) for k in range(10)}, {f"A_{k}": "A" for k in range(10)}
    )
    picked = tract.subset(array, 4, seed=1)
    assert len(picked.units) == 4 and set(picked.units) <= set(array.units)
    dn = tract.tract_array(np.arange(100, 130), 8, seed=2)
    units = np.concatenate(list(dn.units.values()))
    assert units.size == len(set(units.tolist())) == 8 and units.min() >= 100


def test_deltas_compare_seed_means_against_vehicle() -> None:
    per_seed = {
        "vehicle": [{n: {"mfr_hz": 10.0} for n in tract.ARRAYS}] * 2,
        "x": [{n: {"mfr_hz": 20.0} for n in tract.ARRAYS}] * 2,
    }
    got = tract.deltas(per_seed)
    assert got["tract_64"]["x"] == 1.0

"""Stage 4b: lateness, leave-one-out onsets, dose steps and scoring, on synthetic inputs."""

from __future__ import annotations

import pandas as pd

from neuromps.fly import embodied_stage4b as s4b

CONDS = pd.DataFrame(
    {
        "condition": ["vehicle", "f_1", "f_2", "g_1", "g_2"],
        "family": ["vehicle", "f", "f", "g", "g"],
        "dose": [0.0, 0.2, 0.4, 0.5, 1.0],
    }
)


def _medians(rows: dict) -> pd.DataFrame:
    return pd.DataFrame(rows, index=CONDS["condition"])[list(s4b.s4a.MOVEMENTS)]


def test_late_is_strict_and_a_silent_chip_is_late() -> None:
    got = s4b.late({"f": 0.2, "g": 0.5, "h": None}, {"f": 0.4, "g": 0.5, "h": 0.1})
    assert got == {"f": True, "g": False}
    assert s4b.late({"f": 0.2}, {"f": None}) == {"f": True}


def test_within_one_step_uses_each_familys_dose_list() -> None:
    steps = s4b.dose_steps(CONDS)
    assert steps == {"f": [0.2, 0.4], "g": [0.5, 1.0]}
    assert s4b.within_one_step(steps["f"], 0.2, 0.4)
    assert not s4b.within_one_step([0.2, 0.4, 0.6], 0.2, 0.6)
    assert s4b.within_one_step([0.2, 0.4], None, None)
    assert not s4b.within_one_step([0.2, 0.4], 0.2, None)


def test_leave_one_out_drops_a_movement_from_the_index() -> None:
    medians = _medians(
        {
            "proboscis_sugar": [0.3, 0.3, 0.3, 0.3, 0.3],
            "proboscis_sugar_bitter": [0.05, 0.05, 0.05, 0.05, 0.05],
            "reach_depth": [1.0, 1.0, 0.01, 1.0, 1.0],
        }
    )
    assert s4b.movement_onsets(CONDS, medians, list(s4b.s4a.MOVEMENTS)) == {"f": 0.4, "g": None}
    assert s4b.movement_onsets(CONDS, medians, ["proboscis_sugar", "reach_depth"]) == {
        "f": 0.4,
        "g": None,
    }
    assert s4b.movement_onsets(CONDS, medians, ["proboscis_sugar", "proboscis_sugar_bitter"]) == {
        "f": None,
        "g": None,
    }


def test_score_r1_needs_every_family_impaired_and_the_chip_late() -> None:
    medians = _medians(
        {
            "proboscis_sugar": [0.3, 0.02, 0.02, 0.3, 0.02],
            "proboscis_sugar_bitter": [0.05, 0.0, 0.0, 0.05, 0.0],
            "reach_depth": [1.0, 0.05, 0.05, 1.0, 0.05],
        }
    )
    spec = {"predictions": {n: {"test": n} for n in s4b.PREDICTIONS}}
    chip = {"f": 0.4, "g": None}
    got, details = s4b.score(CONDS, medians, spec, chip, previous={"f": 0.2, "g": 1.0})
    assert got["R0_the_movements_are_there"]["outcome"] == "pass"
    assert got["R1_the_chip_is_late_in_every_family"]["outcome"] == "pass"
    assert got["R3_movement_onsets_reproduce"]["outcome"] == "pass"
    assert details["onsets"]["movement"] == {"f": 0.2, "g": 1.0}
    got, _ = s4b.score(CONDS, medians, spec, {"f": 0.2, "g": None}, previous={"f": 0.2, "g": 1.0})
    assert got["R1_the_chip_is_late_in_every_family"]["outcome"] == "fail"


def test_score_is_uninterpretable_without_the_movements() -> None:
    medians = _medians(
        {
            "proboscis_sugar": [0.01] * 5,
            "proboscis_sugar_bitter": [0.0] * 5,
            "reach_depth": [1.0] * 5,
        }
    )
    spec = {"predictions": {n: {"test": n} for n in s4b.PREDICTIONS}}
    got, _ = s4b.score(CONDS, medians, spec, {"f": 0.4, "g": 0.5}, previous={"f": 0.2, "g": 0.5})
    assert got["R0_the_movements_are_there"]["outcome"] == "fail"
    assert {got[n]["outcome"] for n in s4b.PREDICTIONS[1:]} == {"uninterpretable"}


def test_r2_needs_at_least_four_impaired_families() -> None:
    assert s4b.MIN_FAMILIES == 4

"""Stage 4d: conditions, knob mapping, the walking disruption and scoring, on synthetic inputs."""

from __future__ import annotations

import json

import pandas as pd

from neuromps.fly import embodied_stage4d as s4d

SPEC = json.loads(s4d.REGISTRATION.read_text())


def test_conditions_are_the_coarse_and_fine_levels() -> None:
    conds = s4d.conditions().set_index("condition")
    assert len(conds) == 22 + 12
    assert conds.loc["gaba_potentiate_1.7", "knob"] == {"gaba_scale": 1.7}
    assert conds.loc["excitability_block_1.44", "knob"] == {"threshold_scale": 1.44}


def test_knobs_split_into_class_scales_and_a_threshold() -> None:
    assert s4d.knob_parts({"gaba_scale": 0.4}) == ({"gaba": 0.4}, 1.0)
    assert s4d.knob_parts({"threshold_scale": 1.2}) == ({}, 1.2)
    assert s4d.knob_parts({}) == ({}, 1.0)


def test_walking_disruption_is_zero_at_vehicle() -> None:
    got = s4d.walking_disruption(pd.Series({"vehicle": 0.89, "x": 0.44}))
    assert got["vehicle"] == 0.0 and abs(got["x"] - 1.0) < 1e-9


def _inputs(walk_values):
    names = ["vehicle"] + [f"{f}_1" for f in s4d.FAMILIES] + [f"{f}_2" for f in s4d.FAMILIES]
    families = ["vehicle"] + list(s4d.FAMILIES) * 2
    doses = [0.0] + [0.2] * 5 + [0.6] * 5
    conds = pd.DataFrame({"condition": names, "family": families, "dose": doses})
    walk = pd.Series(walk_values, index=names)
    chip = conds.assign(abs_delta_mfr=[0.0] + [0.1] * 5 + [1.0] * 5)
    moves = pd.DataFrame(
        {"proboscis_sugar": 0.3, "proboscis_sugar_bitter": 0.06, "reach_depth": 1.0}, index=names
    )
    return conds, walk, chip, moves


def test_walking_that_fails_first_scores_the_chip_late() -> None:
    conds, walk, chip, moves = _inputs([0.9] + [0.01] * 10)
    got, details = s4d.score(conds, walk, 32, chip, moves, SPEC)
    assert [got[n]["outcome"] for n in s4d.PREDICTIONS] == ["pass"] * 4
    assert details["onsets"]["walking"]["gaba_potentiate"] == 0.2


def test_untouched_walking_leaves_w1_uninterpretable_and_w3_failed() -> None:
    conds, walk, chip, moves = _inputs([0.9] * 11)
    got, _ = s4d.score(conds, walk, 32, chip, moves, SPEC)
    assert got["W1_the_chip_is_late_for_walking"]["outcome"] == "uninterpretable"
    assert got["W2_gaba_potentiation_lateness_is_not_specific_to_feeding"]["outcome"] == "fail"
    assert got["W3_with_walking_the_chip_is_late_in_every_family"]["outcome"] == "fail"


def test_no_rhythm_under_vehicle_makes_the_rest_uninterpretable() -> None:
    conds, walk, chip, moves = _inputs([0.1] * 11)
    got, _ = s4d.score(conds, walk, 32, chip, moves, SPEC)
    assert got["W0_the_walking_rhythm_is_there"]["outcome"] == "fail"
    assert {got[n]["outcome"] for n in s4d.PREDICTIONS[1:]} == {"uninterpretable"}

"""Stage 4c: the fine levels, combined series, resolutions and scoring, on synthetic inputs."""

from __future__ import annotations

import json

import pandas as pd

from neuromps.fly import embodied_stage4c as s4c

MOVES = ["proboscis_sugar", "proboscis_sugar_bitter", "reach_depth"]


def test_levels_hold_vehicle_the_coarse_checks_and_the_fine_grid() -> None:
    spec = json.loads(s4c.REGISTRATION.read_text())
    table = s4c.levels(spec).set_index("condition")
    assert len(table) == 1 + 5 + 12
    assert table.loc["vehicle", "knob"] == {}
    row = table.loc["excitability_block_1.24"]
    assert row["knob"] == {"threshold_scale": 1.24} and abs(row["dose"] - 0.24) < 1e-9
    assert table.loc["gaba_potentiate_2", "knob"] == {"gaba_scale": 2.0}


def test_resolution_names_which_came_first() -> None:
    assert s4c.resolution(0.5, 0.6) == "chip later"
    assert s4c.resolution(0.5, None) == "chip later"
    assert s4c.resolution(0.6, 0.5) == "chip earlier"
    assert s4c.resolution(None, 0.5) == "chip earlier"
    assert s4c.resolution(0.5, 0.5) == "still tied"
    assert s4c.resolution(None, None) == "still tied"


def test_combine_keeps_the_coarse_rows_and_adds_only_new_levels() -> None:
    coarse = pd.DataFrame({"x": [1.0, 2.0]}, index=["vehicle", "f_1"])
    fine = pd.DataFrame({"x": [9.0, 3.0]}, index=["f_1", "f_1.5"])
    got = s4c.combine(coarse, fine)
    assert got.to_dict()["x"] == {"vehicle": 1.0, "f_1": 2.0, "f_1.5": 3.0}


def _inputs():
    conds = pd.DataFrame(
        {
            "condition": ["vehicle", "f_1.2", "f_1.4", "f_1.3"],
            "family": ["vehicle", "f", "f", "f"],
            "dose": [0.0, 0.2, 0.4, 0.3],
        }
    )
    medians = pd.DataFrame(
        {
            "proboscis_sugar": [0.3, 0.3, 0.01, 0.01],
            "proboscis_sugar_bitter": [0.05, 0.05, 0.0, 0.05],
            "reach_depth": [1.0, 1.0, 0.01, 1.0],
        },
        index=conds["condition"],
    )
    chip = pd.DataFrame(
        {
            "condition": conds["condition"],
            "family": conds["family"],
            "dose": conds["dose"],
            "abs_delta_mfr": [0.0, 0.1, 4.0, 0.6],
        }
    )
    return conds, medians, chip


def test_resolve_family_uses_the_combined_series() -> None:
    conds, medians, chip = _inputs()
    got = s4c.resolve_family(conds, medians, chip, "f", MOVES)
    assert got == {"movement_onset": 0.3, "chip_onset": 0.3, "resolution": "still tied"}
    got = s4c.resolve_family(conds, medians, chip, "f", ["proboscis_sugar_bitter"])
    assert got["movement_onset"] == 0.4 and got["resolution"] == "chip earlier"


def test_score_is_uninterpretable_when_the_coarse_levels_do_not_reproduce() -> None:
    conds, medians, chip = _inputs()
    spec = {"predictions": {n: {"test": n} for n in s4c.PREDICTIONS}}
    chips = {"replication": chip, "first_run": chip}
    got, details = s4c.score(spec, conds, medians, chips, 1e-3)
    assert got["F0_coarse_levels_reproduce"]["outcome"] == "fail"
    assert {got[n]["outcome"] for n in s4c.PREDICTIONS[1:]} == {"uninterpretable"}
    assert set(details["resolutions"]) == {"all_three", *(f"without_{m}" for m in MOVES)}
    got, _ = s4c.score(spec, conds, medians, chips, 0.0)
    assert got["F1_gaba_potentiation_chip_later"]["outcome"] == "fail"

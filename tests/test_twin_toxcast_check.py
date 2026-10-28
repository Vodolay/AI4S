"""Tests for the twin-vs-ToxCast positive-control check (synthetic inputs only)."""

import numpy as np
import pandas as pd
import pytest

from neuromps.models import twin_toxcast_check as tc


def test_twin_direction_needs_size_and_consistency():
    assert tc.twin_direction(np.array([1.0] * 8))["direction"] == "up"
    assert tc.twin_direction(np.array([-0.5] * 7 + [0.2]))["direction"] == "down"
    # too small a median: no prediction
    assert tc.twin_direction(np.array([0.05] * 8))["direction"] is None
    # chips disagree: 6 of 8 is below the 7 of 8 rule
    assert tc.twin_direction(np.array([1.0] * 6 + [-1.0] * 2))["direction"] is None
    # too few chips with a defined change
    assert tc.twin_direction(np.array([1.0] * 4 + [np.nan] * 4))["direction"] is None
    # NaN chips are ignored when enough are defined
    got = tc.twin_direction(np.array([1.0] * 6 + [np.nan] * 2))
    assert got["direction"] == "up" and got["n_defined"] == 6


def _wells() -> pd.DataFrame:
    rows = []
    for d in range(12):
        for _v in range(4):  # vehicle wells fall by 50% on every date (log2 -1)
            rows.append({"acid": 2448, "wllt": "n", "spid": "DMSO", "apid": f"d{d}", "rval": -50.0})
        for _p in range(3):  # treated wells stay at baseline: +1 log2 against vehicle
            rows.append(
                {"acid": 2448, "wllt": "p", "spid": "Bicuculline", "apid": f"d{d}", "rval": 0.0}
            )
    rows.append({"acid": 2448, "wllt": "p", "spid": "Bicuculline", "apid": "lonely", "rval": 0.0})
    return pd.DataFrame(rows)


def test_vehicle_corrected_effects_subtract_the_same_date_vehicle_median():
    got = tc.vehicle_corrected(_wells(), 2448, "Bicuculline")
    assert len(got) == 36  # the well on a date without vehicles is dropped
    assert np.allclose(got["effect"], 1.0)


def test_real_effect_is_established_when_the_interval_excludes_zero():
    effects = tc.vehicle_corrected(_wells(), 2448, "Bicuculline")
    effects["effect"] += np.random.default_rng(0).normal(0, 0.1, len(effects))
    got = tc.real_effect(effects, n_resamples=200, seed=1)
    assert got["direction"] == "up" and got["established"]
    assert got["median"] == pytest.approx(1.0, abs=0.1)
    flat = effects.assign(effect=np.random.default_rng(2).normal(0, 1, len(effects)))
    assert not tc.real_effect(flat, n_resamples=200, seed=1)["established"]


def test_real_effect_needs_ten_wells():
    effects = pd.DataFrame({"apid": [f"d{i}" for i in range(9)], "effect": [1.0] * 9})
    assert not tc.real_effect(effects, n_resamples=50, seed=0)["established"]


def test_scoring_counts_only_cells_with_both_directions():
    twin = {"gabaa_block": {"mfr_hz": {"direction": "up"}, "synchrony_index": {"direction": None}}}
    real = {
        "Bicuculline": {
            "mfr_hz": {"direction": "up", "established": True},
            "synchrony_index": {"direction": "up", "established": True},
        }
    }
    drugs = {"Bicuculline": {"knob": "gabaa_block"}}
    cells = tc.score_cells(twin, real, drugs)
    assert cells["Bicuculline"]["mfr_hz"]["agrees"] is True
    assert cells["Bicuculline"]["synchrony_index"]["agrees"] is None
    summary = tc.drug_summary(cells["Bicuculline"])
    assert summary == {"scored": 1, "agreeing": 1}


def test_registered_predictions_pass_and_fail():
    cells = {
        "Bicuculline": {
            f: {"agrees": a}
            for f, a in zip(["mfr_hz", "a", "b", "c"], [True, True, False, True], strict=True)
        },
        "Picrotoxin": {"mfr_hz": {"agrees": False}, "a": {"agrees": True}},
        "Tetrodotoxin": {"mfr_hz": {"agrees": True}, "a": {"agrees": True}, "b": {"agrees": True}},
    }
    got = tc.evaluate_predictions(cells)
    assert got["P1_bicuculline"]["pass"] is True  # 3 of 4
    assert got["P2_picrotoxin"]["pass"] is False  # only 2 scored
    assert got["P3_tetrodotoxin"]["pass"] is True
    assert got["P4_firing_rate"]["pass"] is False  # picrotoxin's firing rate disagrees
    assert {k: v["outcome"] for k, v in got.items()} == {
        "P1_bicuculline": "pass",
        "P2_picrotoxin": "uninterpretable",  # under the three-cell floor
        "P3_tetrodotoxin": "pass",
        "P4_firing_rate": "fail",
    }


def test_follow_firing_rate_scores_established_features_against_the_rate_direction():
    real = {
        "Bicuculline": {
            "mfr_hz": {"direction": "up", "established": True},
            "a": {"direction": "up", "established": True},
            "b": {"direction": "down", "established": True},
            "c": {"direction": "up", "established": False},
        }
    }
    assert tc.follow_firing_rate(real) == {"Bicuculline": {"scored": 2, "agreeing": 1}}

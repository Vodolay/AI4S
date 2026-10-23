"""Scoring rules of the chip-to-animal study, on hand-made numbers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neuromps.fly import analysis, design, direction


def test_the_design_is_read_from_the_registered_file() -> None:
    spec = design.registration()
    conditions = design.conditions(spec)
    assert conditions[0].family == "vehicle" and conditions[0].overrides == {}
    drugs = [c for c in conditions if c.family != "vehicle"]
    assert len(drugs) == 21
    assert {c.family for c in drugs} == {
        "gaba_block",
        "gaba_potentiate",
        "glucl_block",
        "ach_block",
        "excitability_block",
    }
    gaba = [c for c in drugs if c.family == "gaba_block"]
    assert [c.overrides for c in gaba][-1] == {"gaba_scale": 0.0}


def test_dose_is_how_far_the_knob_moved_from_untreated() -> None:
    assert design.Condition("ach_block", 0.2, {"ach_scale": 0.2}).dose == pytest.approx(0.8)
    assert design.Condition("x", 1.6, {"threshold_scale": 1.6}).dose == pytest.approx(0.6)
    assert design.Condition("vehicle", 1.0, {}).dose == 0.0


def test_pair_truth_keeps_well_connected_pairs_once() -> None:
    flow = pd.Series(
        {("A", "B"): 3000, ("B", "A"): 1000, ("A", "C"): 100, ("C", "B"): 5000},
        name="synapses",
    )
    flow.index.names = ["source", "target"]
    pairs = direction.pair_truth(flow, ["A", "B", "C"], min_synapses=2000)
    assert pairs[["a", "b"]].values.tolist() == [["A", "B"], ["B", "C"]]
    assert pairs["asymmetry"].tolist() == pytest.approx([0.5, -1.0])


def test_direction_score_counts_sign_agreement_on_strong_pairs() -> None:
    pairs = pd.DataFrame({"asymmetry": [0.9, -0.8, 0.2, -0.1]})
    measured = np.array([1.0, 0.5, -0.3, -0.2])
    score = direction.score(pairs, measured, strong=0.5)
    assert score["n_pairs"] == 4 and score["n_strong"] == 2
    assert score["sign_agreement"] == pytest.approx(0.5)
    assert score["sign_agreement_strong"] == pytest.approx(0.5)


def test_undefined_measurements_are_left_out_of_the_score() -> None:
    pairs = pd.DataFrame({"asymmetry": [0.9, -0.8, 0.7]})
    score = direction.score(pairs, np.array([1.0, np.nan, 2.0]), strong=0.5)
    assert score["n_pairs"] == 2 and score["sign_agreement_strong"] == 1.0


def test_evoked_response_compares_the_window_after_each_pulse_with_the_one_before() -> None:
    # region 0 fires 5-30 ms after every pulse; region 1 fires only before pulses
    pulses = np.arange(0.2, 2.0, 0.2)
    times = np.concatenate([pulses + 0.010, pulses - 0.010])
    regions = np.concatenate([np.zeros(pulses.size, int), np.ones(pulses.size, int)])
    got = direction.evoked_log_ratio(
        times,
        regions,
        n_regions=2,
        duration_s=2.0,
        period_ms=200.0,
        window_ms=(5, 30),
        reference_ms=(-30, -5),
    )
    assert got[0] > 2 and got[1] < -2


def test_disruption_is_the_mean_absolute_log_change_of_the_four_readouts() -> None:
    vehicle = {"mn9_sugar": 79.0, "mn9_sugar_bitter": 3.0, "abn1_jon_ce": 15.0, "dn_recruited": 1}
    treated = {"mn9_sugar": 159.0, "mn9_sugar_bitter": 3.0, "abn1_jon_ce": 15.0, "dn_recruited": 1}
    assert analysis.disruption(treated, vehicle) == pytest.approx(0.25)


def test_the_sentinel_rule_compares_first_detection_with_first_impairment() -> None:
    rows = pd.DataFrame(
        {
            "family": ["f"] * 3,
            "dose": [0.2, 0.5, 0.8],
            "abs_delta_mfr": [0.1, 0.7, 1.2],
            "disruption": [0.0, 0.2, 0.9],
        }
    )
    got = analysis.sentinel(rows, chip_threshold=0.5, behaviour_threshold=0.5)
    assert got["f"] == {"first_chip_dose": 0.5, "first_behaviour_dose": 0.8, "chip_first": True}
    late = rows.assign(abs_delta_mfr=[0.0, 0.0, 0.0])
    assert analysis.sentinel(late, 0.5, 0.5)["f"]["chip_first"] is False


def test_a_family_that_never_impairs_behaviour_is_not_scored() -> None:
    rows = pd.DataFrame(
        {"family": ["f"], "dose": [0.2], "abs_delta_mfr": [0.9], "disruption": [0.1]}
    )
    assert analysis.sentinel(rows, 0.5, 0.5)["f"]["chip_first"] is None


def test_monotone_trend_includes_the_vehicle_as_dose_zero() -> None:
    assert analysis.trend([0.0, 0.2, 0.4], [1.0, 2.0, 3.0]) == pytest.approx(1.0)

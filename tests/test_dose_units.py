"""Doses go on one scale before anything is ranked against them.

ScreenNeuroPharm records its two lowest bicuculline doses in nM (100 and 300 nM)
and the rest in uM. Three analyses ranked `dose_value` without its unit, so
0.1 uM sat among the highest doses. That hid a dose trend the burst rates carry
(0 of 7 features "moved"; with doses on one scale, 2 do), and moved Task 2's and
HD17's dose correlations. NeuroChip Twin, another entry, found the trend; tracing
the disagreement found the bug.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neuromps.models.dataset import doses_in_um


def test_nanomolar_doses_are_divided_into_micromolar() -> None:
    frame = pd.DataFrame(
        {"dose_value": [100.0, 300.0, 1.0, 30.0], "dose_unit": ["nM", "nM", "uM", "uM"]}
    )
    assert doses_in_um(frame).tolist() == [0.1, 0.3, 1.0, 30.0]


def test_an_untreated_recording_is_dose_zero_whatever_its_unit_field() -> None:
    frame = pd.DataFrame({"dose_value": [None, 1.0], "dose_unit": [None, "uM"]})
    assert doses_in_um(frame).tolist() == [0.0, 1.0]


def test_a_table_without_units_is_read_as_micromolar() -> None:
    """Every source here that records a dose without a unit records it in uM."""
    assert doses_in_um(pd.DataFrame({"dose_value": [1.0, 10.0]})).tolist() == [1.0, 10.0]


def test_an_unknown_unit_is_refused_rather_than_guessed() -> None:
    with pytest.raises(ValueError, match="mg/L"):
        doses_in_um(pd.DataFrame({"dose_value": [1.0], "dose_unit": ["mg/L"]}))


def _bicuculline_like(devices: int = 6) -> pd.DataFrame:
    """A feature that falls with dose, the two lowest doses recorded in nM."""
    doses = [(0.0, None), (100.0, "nM"), (300.0, "nM"), (1.0, "uM"), (10.0, "uM"), (100.0, "uM")]
    rows = []
    for d in range(devices):
        for value, unit in doses:
            um = value / 1000 if unit == "nM" else value
            rate = 20.0 - 4.0 * np.log10(um + 0.01) + 0.01 * d
            rows.append(
                {
                    "device_id": f"D{d}",
                    "dose_value": value or None,
                    "dose_unit": unit,
                    "c0_burst_rate_per_min": rate,
                    "c1_burst_rate_per_min": rate,
                }
            )
    return pd.DataFrame(rows)


def test_the_feature_dose_test_ranks_nanomolar_doses_as_the_lowest() -> None:
    from neuromps.models.dose_response import trends

    out = trends(_bicuculline_like())
    for arm in ("absolute", "baseline_subtracted"):
        entry = out[arm]["by_feature"]["burst_rate_per_min"]
        assert entry["spearman_rho"] < -0.9, (arm, entry)


def test_the_paired_twin_pairs_carry_doses_in_micromolar() -> None:
    from neuromps.twin.delta_inference import real_pairs
    from neuromps.twin.inference import SUMMARY_NAMES

    frame = pd.DataFrame(
        {
            "device_id": ["A", "A", "A"],
            "dose_value": [None, 100.0, 1.0],
            "dose_unit": [None, "nM", "uM"],
            **{name: [1.0, 2.0, 3.0] for name in SUMMARY_NAMES},
        }
    )
    assert sorted(real_pairs(frame)["dose_value"].tolist()) == [0.0, 0.1, 1.0]


def test_task_two_correlates_detection_with_doses_on_one_scale(monkeypatch) -> None:
    """The overall Spearman beside the per-dose table now uses the table's own scale."""
    from neuromps.models import mechanism

    seen = {}

    def capture(doses, probability, **kwargs):
        seen["doses"] = list(np.asarray(doses, dtype=float))
        return 0.0, 1.0

    monkeypatch.setattr(mechanism, "spearmanr", capture)

    class _Model:
        classes_ = np.array(["control", "gabaa_antagonist"])

        def fit(self, x, y):
            return self

        def predict(self, x):
            return np.array(["control"] * len(x))

        def predict_proba(self, x):
            return np.tile([0.5, 0.5], (len(x), 1))

    monkeypatch.setattr(mechanism, "MODELS", {"logistic_regression": _Model()})
    columns = mechanism.delta_columns()
    external = pd.DataFrame(
        {
            "dose_value": [100.0, 300.0, 1.0, 30.0],
            "dose_unit": ["nM", "nM", "uM", "uM"],
            **{c: [0.0] * 4 for c in columns},
        }
    )
    train = pd.DataFrame(
        {"label": ["control", "gabaa_antagonist"], **{c: [0.0, 1.0] for c in columns}}
    )
    mechanism.external_check(train, external, positive="gabaa_antagonist")
    assert seen["doses"] == [0.1, 0.3, 1.0, 30.0]

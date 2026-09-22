"""Paired inference: the pairing, the context vector, and the three checks."""

import numpy as np
import pandas as pd
import pytest

from neuromps.twin.delta_inference import (
    CONTEXT_NAMES,
    DRUG_PARAMETER,
    NEUTRAL_VALUE,
    context_vector,
    dose_trend,
    neutral_recovery,
    real_pairs,
    theta_with_drug,
)
from neuromps.twin.inference import PARAMETER_NAMES, SUMMARY_NAMES


def test_theta_with_drug_places_the_knob_and_leaves_the_rest_in_order() -> None:
    nuisance = np.arange(len(PARAMETER_NAMES) - 1, dtype=float)
    theta = theta_with_drug(nuisance, 0.3)
    assert len(theta) == len(PARAMETER_NAMES)
    assert theta[PARAMETER_NAMES.index(DRUG_PARAMETER)] == 0.3
    others = [v for i, v in enumerate(theta) if PARAMETER_NAMES[i] != DRUG_PARAMETER]
    assert others == list(nuisance)


def test_context_is_the_change_followed_by_the_state_it_started_from() -> None:
    baseline = np.arange(len(SUMMARY_NAMES), dtype=float)
    treated = baseline + 2.0
    context = context_vector(baseline, treated)
    assert len(context) == len(CONTEXT_NAMES) == 2 * len(SUMMARY_NAMES)
    assert np.allclose(context[: len(SUMMARY_NAMES)], 2.0)
    assert np.allclose(context[len(SUMMARY_NAMES) :], baseline)


def test_an_unchanged_recording_has_a_zero_change_half() -> None:
    baseline = np.linspace(0, 1, len(SUMMARY_NAMES))
    assert np.allclose(context_vector(baseline, baseline)[: len(SUMMARY_NAMES)], 0.0)


def _summaries(rows: list[tuple[str, float]]) -> pd.DataFrame:
    """A summary table with one distinguishable feature value per recording."""
    records = []
    for index, (device, dose) in enumerate(rows):
        values = {name: float(index) for name in SUMMARY_NAMES}
        records.append({"device_id": device, "dose_value": dose, **values})
    return pd.DataFrame(records)


def test_real_pairs_gives_each_device_its_own_baseline() -> None:
    pairs = real_pairs(_summaries([("A", 0.0), ("A", 5.0), ("B", 0.0), ("B", 10.0)]))
    assert len(pairs) == 4
    assert set(pairs["device_id"]) == {"A", "B"}
    assert pairs["is_the_baseline_itself"].sum() == 2
    # A device's dosed row must be differenced against its *own* baseline, so
    # the change is not zero and the two devices do not share a reference.
    dosed = pairs[~pairs["is_the_baseline_itself"]]
    for _, row in dosed.iterrows():
        assert not np.allclose(row["context"][: len(SUMMARY_NAMES)], 0.0)


def test_a_device_with_no_untreated_recording_is_dropped_entirely() -> None:
    pairs = real_pairs(_summaries([("A", 0.0), ("A", 5.0), ("B", 10.0), ("B", 20.0)]))
    assert set(pairs["device_id"]) == {"A"}


def test_real_pairs_treats_a_missing_dose_as_untreated() -> None:
    frame = _summaries([("A", 5.0)])
    frame.loc[len(frame)] = {
        **dict.fromkeys(SUMMARY_NAMES, 9.0),
        "device_id": "A",
        "dose_value": None,
    }
    assert len(real_pairs(frame)) == 2


def _pairs_frame(devices: list[str], doses: list[float], baseline_flags: list[bool]):
    return pd.DataFrame(
        {
            "device_id": devices,
            "dose_value": doses,
            "is_the_baseline_itself": baseline_flags,
        }
    )


def test_neutral_recovery_reports_whether_the_interval_covers_one() -> None:
    frame = _pairs_frame(list("ABCD"), [0.0] * 4, [True] * 4)
    medians = [{"median": v} for v in (0.99, 1.0, 1.01, 1.02)]
    out = neutral_recovery(frame, medians)
    assert out["n_untreated"] == 4
    assert out["neutral_value"] == NEUTRAL_VALUE
    assert out["includes_neutral"] is True

    shifted = [{"median": v} for v in (0.30, 0.31, 0.32, 0.33)]
    assert neutral_recovery(frame, shifted)["includes_neutral"] is False


def test_neutral_recovery_declines_to_answer_on_too_few_untreated() -> None:
    frame = _pairs_frame(["A", "B"], [0.0, 5.0], [True, False])
    assert neutral_recovery(frame, [{"median": 1.0}, {"median": 0.5}])["median"] is None


def test_dose_trend_uses_only_the_dosed_rows_and_finds_a_planted_slope() -> None:
    devices = ["A", "A", "A", "B", "B", "B"]
    doses = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0]
    flags = [True, False, False, True, False, False]
    # The knob falls with dose, as a GABA-A antagonist should make it.
    medians = [{"median": v} for v in (1.0, 0.8, 0.6, 1.0, 0.75, 0.55)]
    out = dose_trend(_pairs_frame(devices, doses, flags), medians)
    assert out["n_recordings"] == 4
    assert out["n_devices"] == 2
    assert out["spearman_rho"] < 0


def test_dose_trend_declines_when_every_dosed_row_has_the_same_dose() -> None:
    frame = _pairs_frame(["A"] * 5, [0.0] + [3.0] * 4, [True] + [False] * 4)
    medians = [{"median": v} for v in (1.0, 0.9, 0.8, 0.7, 0.6)]
    assert "note" in dose_trend(frame, medians)


@pytest.mark.parametrize("value", [0.1, 1.0, 2.0])
def test_theta_round_trips_through_the_full_parameter_order(value: float) -> None:
    nuisance = np.zeros(len(PARAMETER_NAMES) - 1)
    theta = theta_with_drug(nuisance, value)
    assert theta[PARAMETER_NAMES.index(DRUG_PARAMETER)] == pytest.approx(value)

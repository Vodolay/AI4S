"""Feature-level dose trends: the check that localises §7.5's failure."""

import numpy as np
import pandas as pd
import pytest

from neuromps.models.dose_response import (
    absolute_values,
    baseline_subtracted,
    chamber_mean,
    holm,
    trends,
    verdict,
)


def _summaries(rows: list[tuple[str, float, float, float]]) -> pd.DataFrame:
    """device, dose, chamber-0 value, chamber-1 value."""
    return pd.DataFrame(
        [
            {"device_id": d, "dose_value": dose, "c0_mfr_hz": a, "c1_mfr_hz": b}
            for d, dose, a, b in rows
        ]
    )


def test_chamber_mean_averages_the_two_chambers() -> None:
    frame = _summaries([("A", 0.0, 2.0, 4.0)])
    assert chamber_mean(frame, "mfr_hz").iloc[0] == 3.0


def test_chamber_mean_ignores_a_chamber_the_summary_left_at_zero() -> None:
    """The summary table writes 0.0 for a missing feature, not NaN."""
    frame = _summaries([("A", 0.0, 0.0, 4.0)])
    assert chamber_mean(frame, "mfr_hz").iloc[0] == 4.0


def test_chamber_mean_is_nan_for_a_feature_the_table_does_not_carry() -> None:
    frame = _summaries([("A", 0.0, 2.0, 4.0)])
    assert np.isnan(chamber_mean(frame, "isi_cv").iloc[0])


def test_baseline_subtraction_uses_each_device_s_own_untreated_recording() -> None:
    frame = _summaries(
        [
            ("A", 0.0, 10.0, 10.0),
            ("A", 5.0, 12.0, 12.0),
            ("B", 0.0, 100.0, 100.0),
            ("B", 5.0, 103.0, 103.0),
        ]
    )
    out = baseline_subtracted(frame, "mfr_hz").set_index("device_id")["value"]
    assert out["A"] == 2.0
    assert out["B"] == 3.0


def test_a_device_without_a_baseline_contributes_nothing() -> None:
    frame = _summaries([("A", 0.0, 10.0, 10.0), ("A", 5.0, 12.0, 12.0), ("B", 5.0, 99.0, 99.0)])
    assert set(baseline_subtracted(frame, "mfr_hz")["device_id"]) == {"A"}


def test_absolute_values_keep_the_baseline_at_dose_zero() -> None:
    frame = _summaries([("A", 0.0, 10.0, 10.0), ("A", 5.0, 12.0, 12.0)])
    out = absolute_values(frame, "mfr_hz")
    assert sorted(out["dose"]) == [0.0, 5.0]
    assert len(out) == 2


def test_holm_is_monotone_and_never_shrinks_a_p_value() -> None:
    adjusted = holm({"a": 0.001, "b": 0.02, "c": 0.5})
    assert adjusted["a"] >= 0.001 and adjusted["b"] >= 0.02 and adjusted["c"] >= 0.5
    assert adjusted["a"] <= adjusted["b"] <= adjusted["c"]
    assert all(v <= 1.0 for v in adjusted.values())


def test_holm_of_one_test_leaves_it_alone() -> None:
    assert holm({"a": 0.03}) == {"a": 0.03}


def _dosed(
    slope: float,
    devices: int = 6,
    doses=(0.0, 1.0, 10.0, 100.0),
    offset_per_device: float = 10.0,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(0)
    for d in range(devices):
        offset = offset_per_device * d
        for dose in doses:
            value = offset + slope * np.log10(dose + 1) + rng.normal(0, 0.01)
            rows.append((f"D{d}", dose, value, value))
    return _summaries(rows)


def test_a_planted_dose_trend_is_found_and_survives_correction() -> None:
    """With the devices on a common baseline, the absolute view sees the trend."""
    out = trends(_dosed(slope=5.0, offset_per_device=0.0))["absolute"]
    entry = out["by_feature"]["mfr_hz"]
    assert entry["spearman_rho"] > 0.5
    assert entry["survives_correction"] is True
    assert "mfr_hz" in out["features_moving_with_dose"]


def test_subtracting_the_baseline_recovers_a_trend_device_spread_had_buried() -> None:
    """Why the module reports both views.

    A between-device offset twice the size of the dose effect dilutes the pooled
    rank correlation to almost nothing; differencing each device against its own
    untreated recording removes the offset and the trend reappears. A null in the
    *absolute* view alone would therefore prove very little.
    """
    frame = _dosed(slope=5.0, offset_per_device=10.0)
    both = trends(frame)
    absolute = both["absolute"]["by_feature"]["mfr_hz"]
    paired = both["baseline_subtracted"]["by_feature"]["mfr_hz"]
    assert absolute["spearman_rho"] < 0.4
    assert paired["spearman_rho"] > 0.9
    assert paired["survives_correction"] is True


def test_a_flat_feature_is_not_reported_as_moving() -> None:
    out = trends(_dosed(slope=0.0))["absolute"]
    assert out["features_moving_with_dose"] == []
    assert out["by_feature"]["mfr_hz"]["survives_correction"] is False


def test_the_verdict_names_the_inverse_model_only_when_features_do_move() -> None:
    moving = {"absolute": {"n_features_moving": 2}, "baseline_subtracted": {"n_features_moving": 0}}
    assert "inverse model" in verdict(moving)["reading"]
    flat = {"absolute": {"n_features_moving": 0}, "baseline_subtracted": {"n_features_moving": 0}}
    assert "no dose signal" in verdict(flat)["reading"]


@pytest.mark.parametrize("builder", [absolute_values, baseline_subtracted])
def test_every_builder_returns_the_three_columns_the_trend_needs(builder) -> None:
    frame = _summaries([("A", 0.0, 1.0, 1.0), ("A", 5.0, 2.0, 2.0)])
    assert set(builder(frame, "mfr_hz").columns) >= {"device_id", "dose", "value"}

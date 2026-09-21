"""Tests for twin calibration: targets from real data, scoring, and the search."""

import numpy as np
import pandas as pd
import pytest

from neuromps.twin.calibrate import (
    CALIBRATION_FEATURES,
    SEARCH_SPACE,
    draw_params,
    real_targets,
    score_features,
)


def _asset(n: int = 12) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "source": "screenneuropharm_bic",
                "data_level": "spike_times",
                "recording.phase": "baseline",
                "qc.usable": True,
                "device.compartment_id": "red" if i % 2 else "green",
                "features.mfr_hz": 10.0 + i * 0.5,
                "features.burst_rate_per_min": 18.0 + i,
                "features.network_burst_rate_per_min": 17.0 + i,
                "features.synchrony_index": 0.7,
                "features.pct_spikes_in_bursts": 90.0,
                "features.active_electrodes": 41,
            }
        )
    # a record that must be ignored: different source, and an exposure phase
    rows.append({**rows[0], "source": "kapucu2022", "features.mfr_hz": 999.0})
    rows.append({**rows[0], "recording.phase": "exposure", "features.mfr_hz": 999.0})
    return pd.DataFrame(rows)


def test_targets_come_from_chamber_baselines_only() -> None:
    targets = real_targets(_asset())
    assert targets["n_records"] == 12
    assert targets["by_feature"]["mfr_hz"]["median"] < 100  # the 999 rows were excluded


def test_targets_carry_a_band_not_just_a_median() -> None:
    targets = real_targets(_asset())
    band = targets["by_feature"]["mfr_hz"]
    assert band["q25"] <= band["median"] <= band["q75"]
    assert set(targets["by_feature"]) == set(CALIBRATION_FEATURES)


def test_score_is_zero_inside_the_band_and_grows_outside() -> None:
    targets = real_targets(_asset())
    inside = {f: targets["by_feature"][f]["median"] for f in CALIBRATION_FEATURES}
    assert score_features(inside, targets)["score"] == pytest.approx(0.0, abs=1e-9)
    far = {f: v * 8 for f, v in inside.items()}
    near = {f: v * 2 for f, v in inside.items()}
    assert score_features(far, targets)["score"] > score_features(near, targets)["score"] > 0


def test_score_penalises_a_missing_feature_rather_than_skipping_it() -> None:
    targets = real_targets(_asset())
    complete = {f: targets["by_feature"][f]["median"] for f in CALIBRATION_FEATURES}
    missing = {**complete, "mfr_hz": None}
    assert score_features(missing, targets)["score"] > score_features(complete, targets)["score"]


def test_score_reports_per_feature_distance() -> None:
    targets = real_targets(_asset())
    features = {f: targets["by_feature"][f]["median"] for f in CALIBRATION_FEATURES}
    out = score_features(features, targets)
    assert set(out["per_feature"]) == set(CALIBRATION_FEATURES)


def test_draws_stay_inside_the_declared_search_space() -> None:
    rng = np.random.default_rng(0)
    for _ in range(25):
        drawn = draw_params(rng)
        for name, (low, high) in SEARCH_SPACE.items():
            assert low <= drawn[name] <= high, name


def test_draws_are_reproducible_from_a_seed() -> None:
    a = [draw_params(np.random.default_rng(7)) for _ in range(3)]
    b = [draw_params(np.random.default_rng(7)) for _ in range(3)]
    assert a == b


def test_search_space_only_names_parameters_that_exist() -> None:
    from neuromps.twin.mea import ElectrodeLayout
    from neuromps.twin.network import NetworkParams

    known = set(NetworkParams().as_record()) | set(ElectrodeLayout().as_record())
    assert set(SEARCH_SPACE) <= known


def test_refinement_stays_near_the_point_and_inside_the_space() -> None:
    from neuromps.twin.calibrate import refine_params

    rng = np.random.default_rng(0)
    centre = {name: (low + high) / 2 for name, (low, high) in SEARCH_SPACE.items()}
    for _ in range(20):
        drawn = refine_params(centre, rng, shrink=0.2)
        for name, (low, high) in SEARCH_SPACE.items():
            width = (high - low) * 0.2
            assert low <= drawn[name] <= high
            assert abs(drawn[name] - centre[name]) <= width / 2 + 1e-9


def test_refinement_clips_at_the_edge_of_the_space() -> None:
    from neuromps.twin.calibrate import refine_params

    rng = np.random.default_rng(1)
    edge = {name: low for name, (low, _) in SEARCH_SPACE.items()}
    drawn = refine_params(edge, rng, shrink=0.5)
    for name, (low, high) in SEARCH_SPACE.items():
        assert low <= drawn[name] <= high


def test_calibrated_params_fall_back_to_defaults_when_absent(tmp_path) -> None:
    from neuromps.twin.calibrate import calibrated_params

    assert calibrated_params(tmp_path / "missing.json") == {}


def test_calibrated_params_are_read_back(tmp_path) -> None:
    import json as _json

    from neuromps.twin.calibrate import calibrated_params

    path = tmp_path / "cal.json"
    path.write_text(_json.dumps({"best": {"draw": {"excitatory_weight": 0.3}}}))
    assert calibrated_params(path) == {"excitatory_weight": 0.3}


def test_calibrated_values_are_inside_the_search_space_they_came_from() -> None:
    """Guards against a stale calibration file written under a different space."""
    from neuromps.twin.calibrate import CALIBRATION_FILE, calibrated_params

    values = calibrated_params(CALIBRATION_FILE)
    for name, value in values.items():
        low, high = SEARCH_SPACE[name]
        assert low <= value <= high, name


def test_real_targets_reports_no_records_when_the_asset_has_no_compartments() -> None:
    """A --quick asset has no device block at all, so the column is absent, not null.

    Zero records is the signal main() uses to skip rather than fit NaN bands.
    """
    quick_asset = _asset().drop(columns=["device.compartment_id"])
    targets = real_targets(quick_asset)
    assert targets["n_records"] == 0
    assert all(band["n"] == 0 for band in targets["by_feature"].values())

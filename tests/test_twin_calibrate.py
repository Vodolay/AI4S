"""Tests for twin calibration: targets from real data, scoring, and the search."""

import numpy as np
import pandas as pd
import pytest

from neuromps.twin.calibrate import (
    CALIBRATION_FEATURES,
    SCORE_SEEDS,
    SEARCH_SPACE,
    boundary_hits,
    draw_params,
    real_targets,
    refine_params,
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
    from neuromps.twin.calibrate import calibrated_params, calibration_file

    values = calibrated_params(calibration_file())
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


def test_a_bounded_feature_is_scored_on_a_logit_distance_not_a_log_ratio() -> None:
    """A log-ratio makes the whole top of a bounded scale look close together.

    The twin sits at synchrony 0.96 against a band ending at 0.84. On a
    log-ratio that is 0.13, less than a rate off by 15%, so the search can buy
    rate accuracy with synchrony error - which is exactly what it did.
    """
    targets = real_targets(_asset())
    band = targets["by_feature"]["synchrony_index"]
    features = {f: targets["by_feature"][f]["median"] for f in CALIBRATION_FEATURES}
    over = {**features, "synchrony_index": 0.964}
    penalty = score_features(over, targets)["per_feature"]["synchrony_index"]
    log_ratio = abs(np.log(0.964 / band["q75"]))
    assert penalty > 5 * log_ratio, "a bounded feature must not be scored as a ratio"


def test_a_bounded_feature_inside_its_band_still_scores_zero() -> None:
    targets = real_targets(_asset())
    features = {f: targets["by_feature"][f]["median"] for f in CALIBRATION_FEATURES}
    out = score_features(features, targets)
    assert out["per_feature"]["synchrony_index"] == 0.0
    assert out["per_feature"]["pct_spikes_in_bursts"] == 0.0


def test_a_bounded_penalty_grows_the_further_outside_the_band_it_sits() -> None:
    targets = real_targets(_asset())
    features = {f: targets["by_feature"][f]["median"] for f in CALIBRATION_FEATURES}
    penalties = [
        score_features({**features, "synchrony_index": v}, targets)["per_feature"][
            "synchrony_index"
        ]
        for v in (0.86, 0.92, 0.99)
    ]
    assert penalties == sorted(penalties)


def test_an_unsynchronised_network_is_scored_rather_than_called_missing() -> None:
    """STTC spans -1 to 1, so 0.0 and below are real answers, not failed runs.

    Scored through the `value <= 0` guard they were indistinguishable from a
    simulation that produced nothing, which is a different thing entirely.
    """
    targets = real_targets(_asset())
    features = {f: targets["by_feature"][f]["median"] for f in CALIBRATION_FEATURES}
    silent = score_features({**features, "synchrony_index": None}, targets)
    asynchronous = score_features({**features, "synchrony_index": 0.0}, targets)
    anticorrelated = score_features({**features, "synchrony_index": -0.4}, targets)
    assert asynchronous["per_feature"]["synchrony_index"] < silent["per_feature"]["synchrony_index"]
    assert (
        anticorrelated["per_feature"]["synchrony_index"]
        > asynchronous["per_feature"]["synchrony_index"]
    )


def test_the_electrode_radius_cannot_exceed_half_the_pitch() -> None:
    """Above pitch/2 two electrodes share neurons and STTC is inflated by construction."""
    from neuromps.twin.mea import ElectrodeLayout

    low, high = SEARCH_SPACE["detection_radius_um"]
    assert high == pytest.approx(ElectrodeLayout().pitch_um / 2)
    assert low < high


def test_boundary_hits_name_a_parameter_pinned_at_an_edge() -> None:
    space = {"a": (0.0, 10.0), "b": (0.0, 10.0), "c": (0.0, 10.0)}
    hits = boundary_hits({"a": 10.0, "b": 5.0, "c": 0.0}, space)
    assert sorted(hits) == ["a", "c"]


def test_boundary_hits_are_empty_for_an_interior_optimum() -> None:
    space = {"a": (0.0, 10.0), "b": (1.0, 2.0)}
    assert boundary_hits({"a": 4.0, "b": 1.5}, space) == []


def test_boundary_hits_ignore_a_parameter_that_was_not_searched() -> None:
    assert boundary_hits({}, {"a": (0.0, 1.0)}) == []


def test_every_search_parameter_is_scored_on_more_than_one_seed() -> None:
    """A single seed lets the search win on a lucky network realisation."""
    assert len(SCORE_SEEDS) > 1
    assert len(set(SCORE_SEEDS)) == len(SCORE_SEEDS)


def test_neuron_count_is_searched_and_drawn_as_a_whole_number() -> None:
    """The twin held 130 neurons/mm2 against 1000-3000 in a real culture.

    It was never searched, so the calibration compensated with an electrode
    radius larger than the electrode pitch.
    """
    assert "n_per_compartment" in SEARCH_SPACE
    rng = np.random.default_rng(0)
    for _ in range(20):
        drawn = draw_params(rng)
        assert isinstance(drawn["n_per_compartment"], int)
        low, high = SEARCH_SPACE["n_per_compartment"]
        assert low <= drawn["n_per_compartment"] <= high


def test_refinement_keeps_the_neuron_count_a_whole_number() -> None:
    rng = np.random.default_rng(0)
    centre = draw_params(np.random.default_rng(1))
    for _ in range(20):
        assert isinstance(refine_params(centre, rng)["n_per_compartment"], int)


def test_a_drawn_neuron_count_overrides_the_default() -> None:
    """Otherwise the search would report a size it never actually simulated."""
    from neuromps.twin.calibrate import simulate_chamber_features

    out = simulate_chamber_features(
        {"n_per_compartment": 60, "detection_radius_um": 100.0}, duration_s=0.2, seed=0
    )
    assert out["params"]["n_per_compartment"] == 60


def test_held_out_seeds_do_not_overlap_the_scoring_seeds() -> None:
    """Picking the best of many draws on two seeds is optimistic; this measures it."""
    from neuromps.twin.calibrate import HELD_OUT_SEEDS

    assert set(HELD_OUT_SEEDS).isdisjoint(SCORE_SEEDS)
    assert len(HELD_OUT_SEEDS) >= 3


def test_held_out_summary_reports_the_optimism_rather_than_hiding_it() -> None:
    from neuromps.twin.calibrate import summarise_held_out

    runs = [
        {"seed": 2, "score": 0.13, "features": {"synchrony_index": 0.78}},
        {"seed": 3, "score": 0.49, "features": {"synchrony_index": 0.66}},
    ]
    out = summarise_held_out(runs, selection_score=0.0756, band=(0.678, 0.843))
    assert out["n_seeds"] == 2
    assert out["score_mean"] == pytest.approx(0.31)
    assert out["selection_score"] == 0.0756
    assert out["optimism"] == pytest.approx(0.31 - 0.0756)
    assert out["synchrony_index_mean"] == pytest.approx(0.72)
    assert out["synchrony_index_in_band_fraction"] == pytest.approx(0.5)


def test_held_out_summary_survives_a_failed_run() -> None:
    from neuromps.twin.calibrate import summarise_held_out

    out = summarise_held_out([], selection_score=0.1, band=(0.678, 0.843))
    assert out["n_seeds"] == 0
    assert out["score_mean"] is None

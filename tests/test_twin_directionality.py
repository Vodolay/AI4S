"""Tests for the HD16 directionality experiment."""

import numpy as np
import pandas as pd
import pytest

from neuromps.twin.directionality import (
    FEATURE_SETS,
    cheapest_configuration,
    discriminate,
    subsample_electrodes,
    summarise,
    sweep,
    truncate,
)
from neuromps.twin.inference import SUMMARY_NAMES


def _heard(seed: int = 0) -> dict[str, np.ndarray]:
    """A full 6x8 recording, named exactly as `electrode_positions` names them."""
    from neuromps.twin.inference import default_layout

    layout = default_layout()
    rng = np.random.default_rng(seed)
    return {
        f"c{c}_r{row}_k{col}": np.sort(rng.uniform(0, 60.0, 200))
        for c in (0, 1)
        for row in range(layout.rows)
        for col in range(layout.cols)
    }


def test_subsampling_keeps_the_requested_count_per_compartment() -> None:
    kept = subsample_electrodes(_heard(), 12)
    assert sum(1 for k in kept if k.startswith("c0_")) == 12
    assert sum(1 for k in kept if k.startswith("c1_")) == 12


def test_subsampling_spans_the_array_instead_of_taking_one_corner() -> None:
    """A corner sample would confound electrode count with chamber coverage."""
    from neuromps.twin.inference import default_layout

    layout = default_layout()
    kept = [k for k in subsample_electrodes(_heard(), 6) if k.startswith("c0_")]
    rows = {int(k.split("_r")[1].split("_k")[0]) for k in kept}
    cols = {int(k.split("_k")[1]) for k in kept}
    assert min(rows) == 0 and max(rows) == layout.rows - 1
    assert min(cols) == 0 and max(cols) == layout.cols - 1


def test_subsampling_asks_for_more_than_exist_returns_all() -> None:
    heard = {k: v for k, v in _heard().items() if k.endswith("_k0")}
    assert len(subsample_electrodes(heard, 48)) == len(heard)


def test_subsampled_trains_are_the_originals_not_copies_with_changed_spikes() -> None:
    heard = _heard()
    kept = subsample_electrodes(heard, 12)
    for name, times in kept.items():
        assert np.array_equal(times, heard[name])


def test_truncation_keeps_only_the_early_spikes() -> None:
    heard = {"c0_e0": np.array([1.0, 5.0, 20.0, 59.0])}
    assert np.array_equal(truncate(heard, 10.0)["c0_e0"], np.array([1.0, 5.0]))
    assert len(truncate(heard, 60.0)["c0_e0"]) == 4


def test_truncation_does_not_mutate_the_input() -> None:
    """The same recording is reused at every duration, so truncation must be pure."""
    heard = {"c0_e0": np.array([1.0, 20.0])}
    truncate(heard, 5.0)
    assert len(heard["c0_e0"]) == 2


def test_summarise_produces_the_simulator_summary_vector() -> None:
    vector = summarise(subsample_electrodes(_heard(), 6), 60.0)
    assert vector.shape == (len(SUMMARY_NAMES),)
    assert np.all(np.isfinite(vector))


def test_feature_sets_only_name_real_summary_dimensions() -> None:
    for name, features in FEATURE_SETS.items():
        assert set(features) <= set(SUMMARY_NAMES), name
        assert features, name


def test_rate_only_and_coupling_only_do_not_overlap() -> None:
    """They must be separable, or the confound check cannot be read."""
    assert not set(FEATURE_SETS["rate_only"]) & set(FEATURE_SETS["coupling_only"])


def _frame(separation: float, n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        symmetric = i % 2 == 0
        row = {name: float(rng.normal(0, 1)) for name in SUMMARY_NAMES}
        row["asymmetry"] = rng.normal(0.0 if symmetric else separation, 1.0)
        rows.append(
            {**row, "symmetric": symmetric, "duration_s": 10.0, "electrodes_per_compartment": 48}
        )
    return pd.DataFrame(rows)


def test_discrimination_of_an_informative_feature_beats_chance() -> None:
    out = discriminate(_frame(4.0), FEATURE_SETS["coupling_only"], n_resamples=200)
    assert out["auc"] > 0.8
    assert out["better_than_chance"] is True


def test_discrimination_of_noise_does_not_claim_a_signal() -> None:
    """An interval that includes 0.5 must not be reported as a result."""
    out = discriminate(_frame(0.0), FEATURE_SETS["coupling_only"], n_resamples=200)
    assert out["auc_ci95"][0] <= 0.5 <= out["auc_ci95"][1]
    assert out["better_than_chance"] is False


def test_discrimination_needs_both_classes() -> None:
    frame = _frame(4.0)
    frame["symmetric"] = True
    assert discriminate(frame, FEATURE_SETS["all"])["auc"] is None


def _cell(duration_s, electrodes, auc, half_width=0.05, feature_set="all", arm="rate_matched"):
    """One sweep row, with an interval around the point estimate."""
    return {
        "feature_set": feature_set,
        "arm": arm,
        "duration_s": duration_s,
        "electrodes_per_compartment": electrodes,
        "auc": auc,
        "auc_ci95": [round(auc - half_width, 3), round(min(auc + half_width, 1.0), 3)],
        "better_than_chance": auc - half_width > 0.5,
    }


def test_cheapest_configuration_prefers_shorter_recordings_over_fewer_electrodes() -> None:
    rows = [_cell(60.0, 6, 0.95), _cell(10.0, 48, 0.90)]
    assert cheapest_configuration(rows, feature_set="all")["duration_s"] == 10.0


def test_cheapest_configuration_needs_the_interval_to_clear_the_target() -> None:
    """A point estimate over the target with a loose interval is not a design rule."""
    rows = [_cell(10.0, 6, 0.85, half_width=0.14)]  # lower bound 0.71, below target
    assert cheapest_configuration(rows, feature_set="all") is None
    assert cheapest_configuration([_cell(10.0, 6, 0.85, half_width=0.02)], feature_set="all")


def test_cheapest_configuration_ignores_cells_that_do_not_reach_the_target() -> None:
    assert cheapest_configuration([_cell(60.0, 48, 0.61)], feature_set="all") is None


def test_cheapest_configuration_reads_the_residualized_set_by_default() -> None:
    """The raw sets can be won by rate imbalance alone, so they are not the headline."""
    rows = [
        _cell(10.0, 6, 0.99, feature_set="all"),
        _cell(60.0, 48, 0.90, feature_set="all_rate_residualized"),
    ]
    chosen = cheapest_configuration(rows)
    assert chosen["feature_set"] == "all_rate_residualized"
    assert chosen["duration_s"] == 60.0


def test_cheapest_configuration_reads_the_rate_matched_arm_by_default() -> None:
    rows = [
        _cell(10.0, 6, 0.99, feature_set="all_rate_residualized", arm="free_rate"),
        _cell(60.0, 48, 0.90, feature_set="all_rate_residualized", arm="rate_matched"),
    ]
    assert cheapest_configuration(rows)["arm"] == "rate_matched"
    assert cheapest_configuration(rows, arm="free_rate")["duration_s"] == 10.0


def test_a_grid_built_for_a_different_question_is_not_reused(tmp_path) -> None:
    """A quick grid must not be inherited by a full run, or vice versa."""
    import json

    from neuromps.twin.directionality import cached_grid, grid_config

    path = tmp_path / "grid.parquet"
    config = grid_config(4, (10.0,), (12, 48), 80)
    pd.DataFrame({"a": [1]}).to_parquet(path, index=False)
    path.with_suffix(".json").write_text(json.dumps(config))
    assert cached_grid(path, config) is not None
    assert cached_grid(path, grid_config(40, (10.0, 30.0, 60.0), (6, 12, 24, 48), 150)) is None


def test_a_grid_without_its_configuration_is_not_reused(tmp_path) -> None:
    from neuromps.twin.directionality import cached_grid, grid_config

    path = tmp_path / "grid.parquet"
    pd.DataFrame({"a": [1]}).to_parquet(path, index=False)
    assert cached_grid(path, grid_config(4, (10.0,), (12,), 80)) is None


def test_the_nonlinear_ceiling_is_scored_alongside_the_linear_baseline() -> None:
    """A linear model failing must not be reported as "the features lack direction"."""
    from neuromps.twin.directionality import SCORED

    assert ("all", "nonlinear", False) in SCORED
    assert ("all", "linear", False) in SCORED


def test_a_nonlinear_rule_is_found_where_a_linear_one_fails() -> None:
    """Both classes centred at zero, differing only in spread: linear cannot separate."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(120):
        symmetric = i % 2 == 0
        row = {name: float(rng.normal(0, 1)) for name in SUMMARY_NAMES}
        row["asymmetry"] = float(rng.normal(0, 0.2 if symmetric else 2.0))
        rows.append({**row, "symmetric": symmetric})
    frame = pd.DataFrame(rows)
    linear = discriminate(frame, FEATURE_SETS["coupling_only"], n_resamples=200, kind="linear")
    nonlinear = discriminate(
        frame, FEATURE_SETS["coupling_only"], n_resamples=200, kind="nonlinear"
    )
    assert linear["auc"] < 0.65
    assert nonlinear["auc"] > linear["auc"]


def test_rate_imbalance_is_reported_per_arm() -> None:
    from neuromps.twin.directionality import rate_imbalance_by_arm

    frame = pd.DataFrame(
        {
            "arm": ["free_rate"] * 4 + ["rate_matched"] * 4,
            "forward_ratio": [1.0, 1.0, 10.0, 10.0] * 2,
            "log_rate_ratio": [0.0, 0.0, -2.0, -2.0, 0.0, 0.0, -0.1, -0.1],
        }
    )
    out = rate_imbalance_by_arm(frame)
    assert out["free_rate"]["10.0"] == pytest.approx(-2.0)
    assert out["rate_matched"]["10.0"] == pytest.approx(-0.1)


def test_sweep_scores_each_arm_separately() -> None:
    frame = pd.concat(
        [_frame(4.0).assign(arm="free_rate"), _frame(0.0).assign(arm="rate_matched")],
        ignore_index=True,
    )
    rows = sweep(frame, n_resamples=100)
    arms = {r["arm"] for r in rows}
    assert arms == {"free_rate", "rate_matched"}


def test_matching_quality_reports_the_worst_rate_gap_per_arm() -> None:
    """A matched arm whose rates still differ must be visibly dismissable."""
    from neuromps.twin.directionality import matching_quality

    frame = pd.DataFrame(
        {
            "arm": ["free_rate"] * 4 + ["rate_matched"] * 4,
            "forward_ratio": [1.0, 1.0, 10.0, 10.0] * 2,
            "c0_mfr_hz": [4.0, 4.0, 2.0, 2.0, 4.0, 4.0, 3.96, 3.96],
            "c1_mfr_hz": [4.0, 4.0, 8.0, 8.0, 4.0, 4.0, 4.04, 4.04],
        }
    )
    out = matching_quality(frame)
    assert out["free_rate"]["worst_relative_rate_gap"] == pytest.approx(1.0)  # 8 vs 4
    assert out["rate_matched"]["worst_relative_rate_gap"] < 0.02


def test_matching_search_returns_both_drive_knobs() -> None:
    """Matching only the target scale shifts absolute rates, which is equally readable."""
    from neuromps.twin.directionality import rate_matching_drives

    drives = rate_matching_drives(
        (1.0, 3.0),
        global_rates=(10.0,),
        target_scales=(1.0, 0.85),
        n_reps=1,
        duration_s=1.0,
        n_per_compartment=40,
    )
    assert set(drives) == {1.0, 3.0}
    for drive in drives.values():
        assert set(drive) == {"background_rate_hz", "background_scale_target"}


def test_residualizing_removes_what_the_rates_explain() -> None:
    """A feature that is a linear function of the rates must carry nothing afterwards."""
    from neuromps.twin.directionality import residualize

    rng = np.random.default_rng(0)
    c0 = rng.uniform(1, 5, 80)
    c1 = rng.uniform(1, 5, 80)
    frame = pd.DataFrame({"c0_mfr_hz": c0, "c1_mfr_hz": c1, "asymmetry": 0.3 * c0 - 0.7 * c1})
    out = residualize(frame, ("asymmetry",))
    assert np.allclose(out["asymmetry"], 0.0, atol=1e-8)


def test_residualizing_keeps_information_the_rates_do_not_explain() -> None:
    from neuromps.twin.directionality import residualize

    rng = np.random.default_rng(1)
    frame = pd.DataFrame(
        {
            "c0_mfr_hz": rng.uniform(1, 5, 80),
            "c1_mfr_hz": rng.uniform(1, 5, 80),
            "asymmetry": rng.normal(0, 1, 80),
        }
    )
    out = residualize(frame, ("asymmetry",))
    assert np.std(out["asymmetry"]) > 0.5 * np.std(frame["asymmetry"])


def test_rate_features_residualized_against_themselves_carry_nothing() -> None:
    """The built-in sanity check: rate_only after residualizing must be at chance."""
    frame = _frame(4.0)
    frame["c0_mfr_hz"] = np.where(frame["symmetric"], 2.0, 6.0)
    frame["c1_mfr_hz"] = np.where(frame["symmetric"], 3.0, 1.0)
    raw = discriminate(frame, FEATURE_SETS["rate_only"], n_resamples=100)
    from neuromps.twin.directionality import residualize

    cleaned = discriminate(
        residualize(frame, FEATURE_SETS["rate_only"]), FEATURE_SETS["rate_only"], n_resamples=100
    )
    assert raw["auc"] > 0.95
    # With no signal left, the point estimate wanders either side of chance on 60
    # samples; what must hold is that it is never reported as better than chance.
    assert cleaned["better_than_chance"] is False
    assert cleaned["auc_ci95"][1] < 0.95


def test_every_scored_combination_is_labelled_distinctly() -> None:
    from neuromps.twin.directionality import SCORED

    labels = {
        (f"{n}_{k}" if k != "linear" else n) + ("_rate_residualized" if r else "")
        for n, k, r in SCORED
    }
    assert len(labels) == len(SCORED)

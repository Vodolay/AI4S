"""Tests for the HD16 directionality experiment."""

import numpy as np
import pandas as pd

from neuromps.twin.directionality import (
    FEATURE_SETS,
    cheapest_configuration,
    discriminate,
    subsample_electrodes,
    summarise,
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


def test_cheapest_configuration_prefers_shorter_recordings_over_fewer_electrodes() -> None:
    rows = [
        {
            "feature_set": "all",
            "duration_s": 60.0,
            "electrodes_per_compartment": 6,
            "auc": 0.9,
            "better_than_chance": True,
        },
        {
            "feature_set": "all",
            "duration_s": 10.0,
            "electrodes_per_compartment": 48,
            "auc": 0.85,
            "better_than_chance": True,
        },
    ]
    assert cheapest_configuration(rows)["duration_s"] == 10.0


def test_cheapest_configuration_ignores_cells_whose_interval_touches_chance() -> None:
    rows = [
        {
            "feature_set": "all",
            "duration_s": 10.0,
            "electrodes_per_compartment": 6,
            "auc": 0.82,
            "better_than_chance": False,
        },
    ]
    assert cheapest_configuration(rows) is None


def test_cheapest_configuration_returns_none_when_nothing_reaches_the_target() -> None:
    rows = [
        {
            "feature_set": "all",
            "duration_s": 60.0,
            "electrodes_per_compartment": 48,
            "auc": 0.61,
            "better_than_chance": True,
        },
    ]
    assert cheapest_configuration(rows) is None


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

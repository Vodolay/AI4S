"""Tests for the cluster bootstrap and for expected calibration error."""

import numpy as np
import pytest

from neuromps.models.bootstrap import (
    DEFAULT_RESAMPLES,
    QUICK_RESAMPLES,
    ci95,
    group_bootstrap_indices,
    group_bootstrap_metric,
)
from neuromps.models.calibration import expected_calibration_error, reliability_bins


def test_resampling_draws_whole_groups_never_partial_ones() -> None:
    """Wells are the unit of independence, so a bootstrap must resample wells."""
    groups = np.array(["a", "a", "a", "b", "b", "c"])
    rng = np.random.default_rng(0)
    for idx in group_bootstrap_indices(groups, rng, n_resamples=20):
        drawn = groups[idx]
        for name, size in zip(*np.unique(groups, return_counts=True), strict=True):
            count = int((drawn == name).sum())
            assert count % size == 0  # whole groups only


def test_resample_count_is_respected() -> None:
    groups = np.array(["a", "b", "c", "d"])
    rng = np.random.default_rng(0)
    assert len(list(group_bootstrap_indices(groups, rng, n_resamples=7))) == 7


def test_bootstrap_is_reproducible_from_a_seed() -> None:
    groups = np.array(list("aabbccdd"))
    values = np.arange(8, dtype=float)
    metric = lambda idx: float(values[idx].mean())  # noqa: E731
    a = group_bootstrap_metric(metric, groups, n_resamples=50, seed=3)
    b = group_bootstrap_metric(metric, groups, n_resamples=50, seed=3)
    assert a == b


def test_interval_brackets_the_point_estimate_for_a_stable_metric() -> None:
    groups = np.array([str(i) for i in range(40)])
    values = np.full(40, 2.0)
    metric = lambda idx: float(values[idx].mean())  # noqa: E731
    out = group_bootstrap_metric(metric, groups, n_resamples=100, seed=0)
    assert out["ci95"][0] == pytest.approx(2.0)
    assert out["ci95"][1] == pytest.approx(2.0)


def test_interval_widens_when_groups_disagree() -> None:
    groups = np.array([str(i) for i in range(20)])
    spread = np.linspace(0.0, 10.0, 20)
    metric = lambda idx: float(spread[idx].mean())  # noqa: E731
    out = group_bootstrap_metric(metric, groups, n_resamples=200, seed=0)
    low, high = out["ci95"]
    assert high - low > 1.0
    assert out["n_resamples"] == 200


def test_ci95_returns_none_for_an_empty_sample() -> None:
    assert ci95([]) == [None, None]


def test_quick_mode_uses_fewer_resamples_than_the_full_run() -> None:
    assert QUICK_RESAMPLES < DEFAULT_RESAMPLES
    assert DEFAULT_RESAMPLES == 1000
    assert QUICK_RESAMPLES == 200


def test_perfect_confidence_and_perfect_accuracy_give_zero_ece() -> None:
    truth = np.array(["a", "b", "a", "b"])
    classes = np.array(["a", "b"])
    probabilities = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    assert expected_calibration_error(truth, probabilities, classes)["ece"] == pytest.approx(0.0)


def test_overconfident_predictions_have_a_large_ece() -> None:
    truth = np.array(["a", "b", "a", "b"])
    classes = np.array(["a", "b"])
    probabilities = np.array([[0.99, 0.01]] * 4)  # always says 'a', always sure
    out = expected_calibration_error(truth, probabilities, classes)
    assert out["ece"] > 0.4


def test_bins_account_for_every_prediction() -> None:
    rng = np.random.default_rng(0)
    classes = np.array(["a", "b", "c"])
    probabilities = rng.dirichlet(np.ones(3), size=60)
    truth = rng.choice(classes, size=60)
    bins = reliability_bins(truth, probabilities, classes, n_bins=10)
    assert sum(b["count"] for b in bins) == 60
    assert all(b["lower"] < b["upper"] for b in bins)


def test_bins_report_confidence_and_accuracy_for_the_diagram() -> None:
    rng = np.random.default_rng(1)
    classes = np.array(["a", "b"])
    probabilities = rng.dirichlet(np.ones(2), size=40)
    truth = rng.choice(classes, size=40)
    for entry in reliability_bins(truth, probabilities, classes, n_bins=5):
        if entry["count"]:
            assert 0.0 <= entry["confidence"] <= 1.0
            assert 0.0 <= entry["accuracy"] <= 1.0

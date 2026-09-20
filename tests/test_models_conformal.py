"""Tests for conformal prediction sets on the mechanism classifier."""

import numpy as np
import pandas as pd
import pytest

from neuromps.models.conformal import (
    conformal_sets,
    coverage_report,
    prediction_sets,
    quantile_threshold,
)
from neuromps.models.dataset import delta_columns


def _pairs(n_per_class: int = 30, spread: float = 0.6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for label, centre in (
        ("control", 0.0),
        ("gabaa_antagonist", 1.5),
        ("na_channel_blocker", -1.5),
    ):
        for i in range(n_per_class):
            rows.append(
                {
                    **{c: rng.normal(centre, spread) for c in delta_columns()},
                    "label": label,
                    "group": f"{label}-{i}",
                    "species": "rat",
                    "source": "kapucu2022",
                }
            )
    return pd.DataFrame(rows)


def test_stricter_alpha_lowers_the_probability_floor() -> None:
    """Demanding more coverage admits labels at lower probability, so sets grow."""
    scores = np.linspace(0, 1, 100)
    strict = quantile_threshold(scores, alpha=0.1)
    loose = quantile_threshold(scores, alpha=0.3)
    assert strict < loose
    assert 0.0 <= strict <= loose <= 1.0


def test_prediction_sets_include_every_label_above_the_threshold() -> None:
    probabilities = np.array([[0.7, 0.2, 0.1], [0.4, 0.35, 0.25]])
    labels = np.array(["a", "b", "c"])
    sets = prediction_sets(probabilities, labels, threshold=0.3)
    assert sets[0] == ["a"]
    assert sets[1] == ["a", "b"]


def test_a_high_floor_can_leave_an_empty_set() -> None:
    """An empty set means no label was confident enough: report it, do not force a guess."""
    probabilities = np.array([[0.4, 0.35, 0.25]])
    sets = prediction_sets(probabilities, np.array(["a", "b", "c"]), threshold=0.7)
    assert sets[0] == []


def test_coverage_is_close_to_the_requested_level() -> None:
    out = conformal_sets(_pairs(), alpha=0.1, seed=0)
    assert 0.80 <= out["empirical_coverage"] <= 1.0
    assert out["alpha"] == 0.1


def test_looser_alpha_gives_smaller_sets_and_less_coverage() -> None:
    strict = conformal_sets(_pairs(), alpha=0.05, seed=0)
    loose = conformal_sets(_pairs(), alpha=0.3, seed=0)
    assert loose["mean_set_size"] <= strict["mean_set_size"]
    assert loose["empirical_coverage"] <= strict["empirical_coverage"]


def test_harder_problems_produce_larger_sets() -> None:
    """Set size is the honest signal: overlapping classes should widen the sets."""
    easy = conformal_sets(_pairs(spread=0.3, seed=1), alpha=0.1, seed=0)
    hard = conformal_sets(_pairs(spread=2.5, seed=1), alpha=0.1, seed=0)
    assert hard["mean_set_size"] > easy["mean_set_size"]


def test_report_counts_empty_and_singleton_sets() -> None:
    out = conformal_sets(_pairs(), alpha=0.1, seed=0)
    assert "fraction_singleton" in out
    assert "fraction_empty" in out
    assert 0.0 <= out["fraction_singleton"] <= 1.0
    assert out["n_test"] > 0


def test_coverage_report_is_per_class_as_well_as_overall() -> None:
    out = conformal_sets(_pairs(), alpha=0.1, seed=0)
    per_class = out["coverage_by_class"]
    assert set(per_class) == {"control", "gabaa_antagonist", "na_channel_blocker"}
    assert all(0.0 <= v <= 1.0 for v in per_class.values())


def test_coverage_report_handles_an_empty_set_without_crashing() -> None:
    truth = np.array(["a", "b"])
    sets = [[], ["a", "b"]]
    out = coverage_report(truth, sets, ["a", "b"])
    assert out["empirical_coverage"] == pytest.approx(0.5)
    assert out["fraction_empty"] == pytest.approx(0.5)


def test_results_are_reproducible() -> None:
    assert conformal_sets(_pairs(), alpha=0.1, seed=3) == conformal_sets(
        _pairs(), alpha=0.1, seed=3
    )

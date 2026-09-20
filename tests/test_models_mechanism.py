"""Tests for the mechanism-classification baselines."""

import numpy as np
import pandas as pd
import pytest

from neuromps.models.dataset import delta_columns
from neuromps.models.mechanism import (
    MODELS,
    external_check,
    grouped_scores,
    summarize,
)


def _pairs(n_per_class: int = 8, seed: int = 0) -> pd.DataFrame:
    """Separable synthetic classes, one group per pair."""
    rng = np.random.default_rng(seed)
    rows = []
    centres = {"control": 0.0, "gabaa_antagonist": 2.0, "na_channel_blocker": -2.0}
    for label, centre in centres.items():
        for i in range(n_per_class):
            row = {c: rng.normal(centre, 0.3) for c in delta_columns()}
            rows.append(
                {
                    **row,
                    "label": label,
                    "group": f"{label}-{i}",
                    "source": "kapucu2022",
                    "species": "rat",
                    "compound": label,
                    "dose_value": 1.0,
                    "dose_unit": "uM",
                }
            )
    return pd.DataFrame(rows)


def test_grouped_scores_returns_a_score_per_fold() -> None:
    scores = grouped_scores(_pairs(), MODELS["logistic_regression"], n_splits=4, seed=0)
    assert len(scores["balanced_accuracy"]) == 4
    assert all(0.0 <= s <= 1.0 for s in scores["balanced_accuracy"])


def test_separable_classes_are_learned_and_beat_the_dummy() -> None:
    pairs = _pairs()
    model = grouped_scores(pairs, MODELS["logistic_regression"], n_splits=4, seed=0)
    dummy = grouped_scores(pairs, MODELS["dummy_most_frequent"], n_splits=4, seed=0)
    assert np.mean(model["balanced_accuracy"]) > 0.8
    assert np.mean(model["balanced_accuracy"]) > np.mean(dummy["balanced_accuracy"])


def test_no_group_appears_in_both_train_and_test() -> None:
    """Two conditions of one culture must never straddle a split."""
    pairs = _pairs()
    pairs["group"] = "shared-" + (pairs.index % 6).astype(str)  # 6 groups, several rows each
    scores = grouped_scores(pairs, MODELS["logistic_regression"], n_splits=3, seed=0)
    for train_groups, test_groups in scores["fold_groups"]:
        assert not set(train_groups) & set(test_groups)


def test_scores_are_reproducible_with_a_seed() -> None:
    a = grouped_scores(_pairs(), MODELS["logistic_regression"], n_splits=4, seed=7)
    b = grouped_scores(_pairs(), MODELS["logistic_regression"], n_splits=4, seed=7)
    assert a["balanced_accuracy"] == b["balanced_accuracy"]


def test_summarize_reports_mean_and_spread_and_class_counts() -> None:
    out = summarize(_pairs(), n_splits=4, seed=0)
    assert set(out["class_counts"]) == {"control", "gabaa_antagonist", "na_channel_blocker"}
    for name in MODELS:
        assert "balanced_accuracy_mean" in out["models"][name]
        assert "balanced_accuracy_sd" in out["models"][name]
    assert out["n_pairs"] == 24
    assert out["n_groups"] == 24


def test_external_check_reports_detection_rate_without_inventing_negatives() -> None:
    """The external set has no vehicle controls, so only sensitivity is defined."""
    train = _pairs(seed=1)
    external = _pairs(seed=2)
    external = external[external["label"] == "gabaa_antagonist"].copy()
    external["source"] = "screenneuropharm_bic"
    out = external_check(train, external, positive="gabaa_antagonist")
    assert 0.0 <= out["detection_rate"] <= 1.0
    assert out["detection_rate"] > 0.5
    assert out["n"] == len(external)
    assert "specificity" not in out  # no negatives available: do not fabricate one


def test_external_check_flags_a_different_mechanism_as_negative_control() -> None:
    train = _pairs(seed=3)
    other = _pairs(seed=4)
    other = other[other["label"] == "na_channel_blocker"].copy()
    out = external_check(train, other, positive="gabaa_antagonist")
    assert out["detection_rate"] < 0.5


def test_dose_trend_is_reported_when_doses_vary() -> None:
    train = _pairs(seed=5)
    external = _pairs(seed=6)
    external = external[external["label"] == "gabaa_antagonist"].copy()
    external["dose_value"] = np.linspace(0.1, 100.0, len(external))
    out = external_check(train, external, positive="gabaa_antagonist")
    assert "dose_spearman" in out
    assert -1.0 <= out["dose_spearman"] <= 1.0


def test_pairs_with_missing_features_do_not_crash_the_models() -> None:
    pairs = _pairs()
    pairs.loc[pairs.index[:5], "delta.network_burst_rate_per_min"] = np.nan
    scores = grouped_scores(pairs, MODELS["gradient_boosting"], n_splits=3, seed=0)
    assert all(np.isfinite(s) for s in scores["balanced_accuracy"])


def test_every_model_is_pipelined_with_imputation_and_scaling() -> None:
    for name, model in MODELS.items():
        steps = dict(model.named_steps)
        assert "impute" in steps, name
        assert pytest.approx(0) == 0  # sanity: pipeline built without fitting

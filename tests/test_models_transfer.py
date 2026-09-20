"""Tests for the rat -> human transfer study and its domain adaptation."""

import numpy as np
import pandas as pd
import pytest

from neuromps.models.dataset import delta_columns
from neuromps.models.transfer import (
    coral,
    per_species_standardize,
    species_shift,
    transfer_scores,
)


def _pairs(species: str, shift: float, n: int = 20, seed: int = 0) -> pd.DataFrame:
    """Two separable classes, offset by a species-specific shift."""
    rng = np.random.default_rng(seed)
    rows = []
    for label, centre in (("control", 0.0), ("gabaa_antagonist", 2.0)):
        for i in range(n):
            row = {c: rng.normal(centre + shift, 0.4) for c in delta_columns()}
            rows.append(
                {
                    **row,
                    "label": label,
                    "species": species,
                    "source": "kapucu2022",
                    "group": f"{species}-{label}-{i}",
                    "compound": label,
                    "dose_value": 1.0,
                    "dose_unit": "uM",
                }
            )
    return pd.DataFrame(rows)


def test_coral_aligns_source_covariance_to_target() -> None:
    rng = np.random.default_rng(0)
    source = rng.normal(0, 1, size=(300, 4))
    target = rng.normal(0, 1, size=(300, 4)) @ np.diag([3.0, 0.5, 2.0, 1.0])
    aligned = coral(source, target)
    before = np.linalg.norm(np.cov(source.T) - np.cov(target.T))
    after = np.linalg.norm(np.cov(aligned.T) - np.cov(target.T))
    assert after < before / 2


def test_coral_leaves_matched_distributions_almost_unchanged() -> None:
    rng = np.random.default_rng(1)
    source = rng.normal(0, 1, size=(200, 3))
    target = rng.normal(0, 1, size=(200, 3))
    assert np.allclose(coral(source, target), source, atol=0.6)


def test_per_species_standardization_removes_the_offset() -> None:
    rat = _pairs("rat", shift=0.0, seed=2)
    human = _pairs("human", shift=5.0, seed=3)
    both = pd.concat([rat, human], ignore_index=True)
    out = per_species_standardize(both)
    means = out.groupby("species")[delta_columns()].mean()
    assert np.allclose(means.to_numpy(), 0.0, atol=1e-6)


def test_species_shift_reports_per_feature_difference_and_ranks_invariance() -> None:
    rat = _pairs("rat", shift=0.0, seed=4)
    human = _pairs("human", shift=1.0, seed=5)
    shift = species_shift(pd.concat([rat, human], ignore_index=True))
    assert set(shift["by_feature"]) == set(f.removeprefix("delta.") for f in delta_columns())
    first = shift["most_invariant_features"][0]
    worst = shift["most_shifted_features"][0]
    assert abs(shift["by_feature"][first]["standardized_difference"]) <= abs(
        shift["by_feature"][worst]["standardized_difference"]
    )


def test_transfer_scores_report_both_directions_and_a_within_species_baseline() -> None:
    rat = _pairs("rat", shift=0.0, seed=6)
    human = _pairs("human", shift=0.0, seed=7)
    out = transfer_scores(pd.concat([rat, human], ignore_index=True), seed=0)
    assert "rat_to_human" in out and "human_within" in out
    assert 0.0 <= out["rat_to_human"]["balanced_accuracy"] <= 1.0
    assert out["rat_to_human"]["n_train"] == len(rat)
    assert out["rat_to_human"]["n_test"] == len(human)


def test_transfer_degrades_when_the_species_differ_and_adaptation_helps() -> None:
    """A pure offset between species is exactly what standardization should fix."""
    rat = _pairs("rat", shift=0.0, seed=8)
    human = _pairs("human", shift=6.0, seed=9)
    both = pd.concat([rat, human], ignore_index=True)
    plain = transfer_scores(both, seed=0)["rat_to_human"]["balanced_accuracy"]
    adapted = transfer_scores(both, adapt="standardize", seed=0)["rat_to_human"][
        "balanced_accuracy"
    ]
    assert adapted > plain


def test_adaptation_options_are_reported_not_silently_chosen() -> None:
    both = pd.concat(
        [_pairs("rat", 0.0, seed=10), _pairs("human", 2.0, seed=11)], ignore_index=True
    )
    for adapt in (None, "standardize", "coral"):
        out = transfer_scores(both, adapt=adapt, seed=0)
        assert out["adaptation"] == (adapt or "none")


def test_scores_are_reproducible() -> None:
    both = pd.concat(
        [_pairs("rat", 0.0, seed=12), _pairs("human", 1.0, seed=13)], ignore_index=True
    )
    a = transfer_scores(both, seed=3)
    b = transfer_scores(both, seed=3)
    assert a["rat_to_human"] == b["rat_to_human"]


def test_bootstrap_interval_is_reported_for_the_transfer_score() -> None:
    both = pd.concat(
        [_pairs("rat", 0.0, seed=14), _pairs("human", 0.0, seed=15)], ignore_index=True
    )
    out = transfer_scores(both, seed=0)["rat_to_human"]
    low, high = out["balanced_accuracy_ci95"]
    assert 0.0 <= low <= out["balanced_accuracy"] <= high <= 1.0


def test_classes_absent_from_the_target_species_are_excluded() -> None:
    """A class the target species never received cannot be scored, so drop it explicitly."""
    rat = _pairs("rat", 0.0, seed=16)
    human = _pairs("human", 0.0, seed=17)
    human = human[human["label"] == "control"]
    out = transfer_scores(pd.concat([rat, human], ignore_index=True), seed=0)
    assert out["shared_labels"] == ["control"]
    assert out["rat_to_human"] is None  # degenerate: say so instead of scoring it
    assert "cannot be scored" in out["note"]


def test_three_shared_classes_are_scored_normally() -> None:
    rat = _pairs("rat", 0.0, seed=18)
    human = _pairs("human", 0.0, seed=19)
    out = transfer_scores(pd.concat([rat, human], ignore_index=True), seed=0)
    assert out["rat_to_human"]["balanced_accuracy"] == pytest.approx(1.0, abs=0.4)

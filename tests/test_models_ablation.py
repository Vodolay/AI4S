"""Tests for feature-subset ablations."""

import numpy as np
import pandas as pd
import pytest

from neuromps.models.ablation import FEATURE_SUBSETS, ablate, leave_one_out
from neuromps.models.dataset import delta_columns


def _pairs(informative: str, n: int = 24, seed: int = 0) -> pd.DataFrame:
    """Only `informative` separates the classes; every other feature is noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for label, centre in (("control", 0.0), ("gabaa_antagonist", 4.0)):
        for i in range(n):
            row = {c: rng.normal(0.0, 0.5) for c in delta_columns()}
            row[f"delta.{informative}"] = rng.normal(centre, 0.4)
            rows.append(
                {
                    **row,
                    "label": label,
                    "group": f"{label}-{i}",
                    "species": "rat",
                    "source": "kapucu2022",
                }
            )
    return pd.DataFrame(rows)


def test_subsets_are_named_and_within_the_feature_vector() -> None:
    known = {c.removeprefix("delta.") for c in delta_columns()}
    for name, features in FEATURE_SUBSETS.items():
        assert set(features) <= known, name


def test_ablation_scores_every_subset() -> None:
    out = ablate(_pairs("mfr_hz"), seed=0)
    assert set(out) == set(FEATURE_SUBSETS)
    for name, result in out.items():
        assert 0.0 <= result["balanced_accuracy_mean"] <= 1.0, name
        assert result["n_features"] == len(FEATURE_SUBSETS[name])


def test_a_subset_missing_the_informative_feature_scores_near_chance() -> None:
    out = ablate(_pairs("synchrony_index"), seed=0)
    assert out["rate_only"]["balanced_accuracy_mean"] < 0.7
    assert out["synchrony_only"]["balanced_accuracy_mean"] > 0.8


def test_leave_one_out_finds_the_feature_that_matters() -> None:
    out = leave_one_out(_pairs("mfr_hz"), seed=0)
    dropped = min(out, key=lambda f: out[f]["balanced_accuracy_mean"])
    assert dropped == "mfr_hz"
    assert out["mfr_hz"]["delta_vs_all"] < 0  # removing it costs accuracy


def test_leave_one_out_reports_change_against_the_full_vector() -> None:
    out = leave_one_out(_pairs("burst_rate_per_min"), seed=0)
    assert all("delta_vs_all" in v for v in out.values())
    assert len(out) == len(delta_columns())


def test_ablation_is_reproducible() -> None:
    pairs = _pairs("isi_cv")
    assert ablate(pairs, seed=1) == ablate(pairs, seed=1)


def test_ablation_handles_a_subset_with_one_feature() -> None:
    out = ablate(_pairs("mfr_hz"), seed=0)
    assert out["mfr_only"]["n_features"] == 1
    assert out["mfr_only"]["balanced_accuracy_mean"] == pytest.approx(1.0, abs=0.2)


def test_species_invariant_subset_matches_the_transfer_study() -> None:
    """The subset named after the HD9 result must contain exactly those features."""
    assert set(FEATURE_SUBSETS["species_invariant"]) == {
        "mfr_hz",
        "pct_spikes_in_bursts",
        "burst_rate_per_min",
    }


def test_np_isfinite_guard_for_degenerate_subsets() -> None:
    pairs = _pairs("mfr_hz")
    pairs[delta_columns()] = np.nan
    out = ablate(pairs, seed=0)
    assert all(np.isfinite(v["balanced_accuracy_mean"]) for v in out.values())

"""Every headline metric must ship an interval; these tests fail if one loses it.

Two levels: the functions that produce the numbers, and the written
`outputs/metrics.json`. The file-level test skips when the pipeline has not been
run, so a fresh clone is not blocked, but it will fail loudly once metrics exist
without intervals.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neuromps.models.ablation import ablate, leave_one_out
from neuromps.models.dataset import delta_columns
from neuromps.models.mechanism import summarize
from neuromps.models.transfer import species_shift

METRICS = Path(__file__).resolve().parents[1] / "outputs" / "metrics.json"
QUICK = {"n_resamples_quick": 200}


def _pairs(n: int = 10, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for species in ("rat", "human"):
        for label, centre in (("control", 0.0), ("gabaa_antagonist", 2.0)):
            for i in range(n):
                rows.append(
                    {
                        **{c: rng.normal(centre, 0.5) for c in delta_columns()},
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


def test_every_model_reports_a_balanced_accuracy_interval() -> None:
    out = summarize(_pairs(), seed=0, quick=True)
    assert out["bootstrap_resamples"] == QUICK["n_resamples_quick"]
    for name, result in out["models"].items():
        assert "balanced_accuracy_ci95" in result, name
        low, high = result["balanced_accuracy_ci95"]
        assert low is not None and high is not None, name
        assert low <= result["balanced_accuracy_mean"] + 1e-9
        assert result["balanced_accuracy_mean"] - 1e-9 <= high


def test_every_species_shift_feature_reports_an_interval() -> None:
    shift = species_shift(_pairs(), quick=True)
    assert shift["bootstrap_resamples"] == QUICK["n_resamples_quick"]
    for feature, result in shift["by_feature"].items():
        assert "difference_ci95" in result, feature
        assert result["difference_ci95"][0] is not None, feature


def test_every_feature_subset_reports_an_interval() -> None:
    for name, result in ablate(_pairs(), seed=0, quick=True).items():
        assert "balanced_accuracy_ci95" in result, name
        assert result["bootstrap_resamples"] == QUICK["n_resamples_quick"]


def test_every_leave_one_out_entry_reports_an_interval() -> None:
    for feature, result in leave_one_out(_pairs(), seed=0, quick=True).items():
        assert "balanced_accuracy_ci95" in result, feature


def _written() -> dict:
    if not METRICS.exists():
        pytest.skip("outputs/metrics.json absent; run python run_all.py --stage models")
    return json.loads(METRICS.read_text())


def test_written_task1_has_calibration_and_intervals() -> None:
    task1 = _written().get("mechanism_baselines", {}).get("task1_within_platform")
    if task1 is None:
        pytest.skip("mechanism baselines not in metrics.json yet")
    assert "calibration" in task1
    assert "ece" in task1["calibration"]
    assert task1["calibration"]["bins"], "reliability bins must be recorded"
    assert sum(b["count"] for b in task1["calibration"]["bins"]) == task1["calibration"]["n"]
    for name, model in task1["models"].items():
        assert model.get("balanced_accuracy_ci95", [None])[0] is not None, name


def test_written_species_shift_and_ablations_have_intervals() -> None:
    written = _written()
    shift = written.get("species_transfer", {}).get("species_shift")
    if shift is not None:
        for feature, result in shift["by_feature"].items():
            assert result.get("difference_ci95", [None])[0] is not None, feature
    ablations = written.get("uncertainty_and_ablations", {}).get("feature_subsets")
    if ablations is not None:
        for name, result in ablations.items():
            assert result.get("balanced_accuracy_ci95", [None])[0] is not None, name


def test_written_metrics_record_the_resample_count() -> None:
    written = _written()
    task1 = written.get("mechanism_baselines", {}).get("task1_within_platform")
    if task1 is None:
        pytest.skip("mechanism baselines not in metrics.json yet")
    assert task1["bootstrap_resamples"] in (200, 1000)

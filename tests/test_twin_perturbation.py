"""Tests for the drug-knob validation: simulated direction vs real direction."""

import numpy as np
import pandas as pd
import pytest

from neuromps.models.dataset import delta_columns
from neuromps.twin.perturbation import (
    KNOBS,
    agreement_table,
    real_directions,
    simulated_delta,
)


def _pairs(mechanism: str, deltas: dict[str, float], n: int = 12, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        row = {c: 0.0 for c in delta_columns()}
        for feature, value in deltas.items():
            row[f"delta.{feature}"] = value + rng.normal(0, 0.05)
        rows.append({**row, "label": mechanism, "group": f"g{i}", "source": "x", "species": "rat"})
    return pd.DataFrame(rows)


def test_real_direction_is_read_from_the_asset_not_hardcoded() -> None:
    pairs = _pairs("gabaa_antagonist", {"mfr_hz": -1.0, "synchrony_index": 0.8})
    out = real_directions(pairs, "gabaa_antagonist", n_resamples=200, seed=0)
    assert out["n_pairs"] == 12
    assert out["by_feature"]["mfr_hz"]["median_delta"] < 0
    assert out["by_feature"]["synchrony_index"]["median_delta"] > 0


def test_a_real_direction_counts_as_established_only_if_its_interval_excludes_zero() -> None:
    clear = real_directions(_pairs("x", {"mfr_hz": -1.5}), "x", n_resamples=200, seed=0)
    assert clear["by_feature"]["mfr_hz"]["established"] is True
    # a feature left at zero with noise should not count as a direction
    assert clear["by_feature"]["isi_cv"]["established"] is False


def test_missing_mechanism_returns_no_directions_rather_than_failing() -> None:
    out = real_directions(_pairs("gabaa_antagonist", {"mfr_hz": -1.0}), "absent", n_resamples=50)
    assert out["n_pairs"] == 0
    assert out["by_feature"] == {}


def test_knobs_cover_the_mechanisms_the_asset_labels() -> None:
    mechanisms = {knob["mechanism"] for knob in KNOBS.values()}
    assert {"gabaa_antagonist", "gabaa_agonist_or_pam", "na_channel_blocker"} <= mechanisms
    for name, knob in KNOBS.items():
        assert knob["parameter"] in ("gabaa_scale", "ampa_scale", "threshold_scale"), name
        assert knob["value"] != 1.0, name  # 1.0 is untreated: a knob must move


def test_agreement_marks_matching_signs_and_flags_mismatches() -> None:
    real = {
        "by_feature": {
            "mfr_hz": {"median_delta": -1.0, "established": True},
            "synchrony_index": {"median_delta": 0.5, "established": True},
            "isi_cv": {"median_delta": 0.2, "established": False},
        }
    }
    simulated = {"mfr_hz": -0.4, "synchrony_index": -0.3, "isi_cv": 0.9}
    table = agreement_table(simulated, real)
    assert table["mfr_hz"]["agrees"] is True
    assert table["synchrony_index"]["agrees"] is False
    assert table["isi_cv"]["agrees"] is None  # real direction not established
    assert table["mfr_hz"]["real_direction"] == "down"
    assert table["synchrony_index"]["simulated_direction"] == "down"


def test_agreement_summary_counts_only_established_features() -> None:
    real = {
        "by_feature": {
            "a": {"median_delta": 1.0, "established": True},
            "b": {"median_delta": -1.0, "established": True},
            "c": {"median_delta": 1.0, "established": False},
        }
    }
    table = agreement_table({"a": 0.5, "b": -0.5, "c": -9.0}, real)
    scored = [v["agrees"] for v in table.values() if v["agrees"] is not None]
    assert scored == [True, True]


def test_simulated_delta_is_a_log_ratio_against_the_simulated_baseline() -> None:
    baseline = {"mfr_hz": 4.0, "synchrony_index": 0.4}
    treated = {"mfr_hz": 2.0, "synchrony_index": 0.8}
    out = simulated_delta(treated, baseline)
    assert out["mfr_hz"] == pytest.approx(-1.0)
    assert out["synchrony_index"] == pytest.approx(1.0)


def test_simulated_delta_is_undefined_without_a_baseline_value() -> None:
    assert simulated_delta({"mfr_hz": 2.0}, {"mfr_hz": None})["mfr_hz"] is None
    assert simulated_delta({"mfr_hz": None}, {"mfr_hz": 2.0})["mfr_hz"] is None


def test_a_silenced_network_reads_as_abolished_not_unknown() -> None:
    """A blocker that stops all firing must give a direction, not a missing value."""
    from neuromps.models.dataset import FLOOR
    from neuromps.twin.perturbation import ZERO_WHEN_SILENT

    baseline = {f: 4.0 for f in ZERO_WHEN_SILENT}
    silent = {f: 0.0 for f in ZERO_WHEN_SILENT}
    out = simulated_delta(silent, baseline)
    for feature in ZERO_WHEN_SILENT:
        assert out[feature] == pytest.approx(np.log2(FLOOR), abs=1e-3)  # same floor as the real ETL
        assert out[feature] < 0

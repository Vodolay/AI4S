"""The registered logISI follow-up (D34): S4 = the ladder's S3 with the logISI detector."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_extractor_ladder import _write_plate

from neuromps.models.extractor_agreement import paired_table
from neuromps.models.extractor_ladder import ALL_DEFINITIONS, LADDER, changes_from
from neuromps.models.extractor_logisi import (
    LOGISI,
    MOVE,
    REGISTRATION,
    VARIANTS,
    fallback_record,
    moved_pairs,
    plot_logisi,
    predictions,
    results_text,
    threshold_distribution,
    thresholds_table,
)

PLATES3 = ("hPSC_MEA1", "hPSC_MEA2", "Rat_MEA1")


def test_s4_is_s3_with_the_logisi_detector_only() -> None:
    assert LOGISI.name == "S4_logisi"
    assert changes_from(LADDER[3], LOGISI) == ["burst_detector"]
    assert LOGISI.params.burst_detector == "logisi"
    assert LOGISI.params.burst_min_spikes == 5
    assert LOGISI.params.logisi_merge_s == pytest.approx(0.1)
    assert VARIANTS == ("S0_ours", "S3_rate_per_bursting_electrode", "S4_logisi")


def test_the_ladder_definitions_do_not_include_s4() -> None:
    """The ladder's own numbers must not change: S4 is reported beside it, not inside it."""
    assert LOGISI.name not in {d.name for d in ALL_DEFINITIONS}


def test_registration_names_the_metrics_key_and_files() -> None:
    reg = json.loads(REGISTRATION.read_text())
    assert reg["outputs"]["metrics_key"] == "extractor_agreement.logisi"
    assert set(reg["predictions"]) == {
        "L1_burst_duration_closes",
        "L2_spikes_per_burst_closes",
        "L3_hpsc_burst_rate_closes",
        "L4_rat_not_worse",
    }


def _stats(values: dict[tuple[str, str], tuple[float, float]]) -> dict:
    """S4 statistics: (scope, feature) -> (median ratio, rho)."""
    out: dict = {}
    for (scope, feature), (ratio, rho) in values.items():
        out.setdefault(scope, {})[feature] = {"median_ratio": ratio, "spearman_rho": rho}
    return out


ALL_PASS = {
    ("pooled", "burst_duration_s"): (1.0, 0.95),
    ("pooled", "spikes_per_burst"): (1.02, 0.97),
    ("hPSC_MEA1", "burst_rate_per_min"): (0.96, 0.9),
    ("hPSC_MEA2", "burst_rate_per_min"): (1.04, 0.9),
    ("Rat_MEA1", "burst_rate_per_min"): (1.0, 0.9),
    ("Rat_MEA1", "burst_duration_s"): (0.95, 0.9),
}
NO_FALLBACK = {p: {"n_active": 100, "n_fallback": 10} for p in PLATES3}


def test_predictions_pass_inside_their_bands() -> None:
    got = predictions(_stats(ALL_PASS), NO_FALLBACK, PLATES3)
    assert [p["outcome"] for p in got.values()] == ["pass"] * 4
    assert all(p["pass"] is True for p in got.values())


def test_predictions_fail_on_a_band_or_on_rho() -> None:
    values = dict(ALL_PASS)
    values[("pooled", "burst_duration_s")] = (1.0, 0.89)  # rho below 0.90
    values[("hPSC_MEA2", "burst_rate_per_min")] = (1.06, 0.9)  # ratio above 1.05
    got = predictions(_stats(values), NO_FALLBACK, PLATES3)
    assert got["L1_burst_duration_closes"]["outcome"] == "fail"
    assert got["L3_hpsc_burst_rate_closes"]["outcome"] == "fail"
    assert got["L2_spikes_per_burst_closes"]["outcome"] == "pass"
    assert got["L1_burst_duration_closes"]["pass"] is False


def test_predictions_are_uninterpretable_above_half_fallback_on_their_plates() -> None:
    """Rat at 90% fallback: L4 and the pooled L1, L2 cannot be read; L3 (hPSC, 35%) can."""
    fallback = {
        "hPSC_MEA1": {"n_active": 100, "n_fallback": 0},
        "hPSC_MEA2": {"n_active": 100, "n_fallback": 70},
        "Rat_MEA1": {"n_active": 100, "n_fallback": 90},
    }
    got = predictions(_stats(ALL_PASS), fallback, PLATES3)
    assert got["L4_rat_not_worse"]["outcome"] == "uninterpretable"
    assert got["L3_hpsc_burst_rate_closes"]["outcome"] == "pass"
    # Pooled over the three plates: 160 of 300 fall back.
    assert got["L1_burst_duration_closes"]["outcome"] == "uninterpretable"
    assert got["L1_burst_duration_closes"]["fallback_share"] == pytest.approx(160 / 300, abs=1e-4)
    assert got["L3_hpsc_burst_rate_closes"]["fallback_share"] == pytest.approx(0.35)
    assert got["L4_rat_not_worse"]["pass"] is False


def test_predictions_are_uninterpretable_when_a_plate_is_missing() -> None:
    got = predictions(_stats(ALL_PASS), NO_FALLBACK, ("hPSC_MEA1",))
    assert got["L4_rat_not_worse"]["outcome"] == "uninterpretable"
    assert got["L3_hpsc_burst_rate_closes"]["outcome"] == "uninterpretable"


def _pairs(rows: list[tuple[str, str, float, float]]) -> pd.DataFrame:
    out = []
    for i, (plate, feature, s3, s4) in enumerate(rows):
        for variant, value in (("S3_rate_per_bursting_electrode", s3), ("S4_logisi", s4)):
            out.append(
                {
                    "variant": variant,
                    "plate": plate,
                    "well": f"W{i}",
                    "div": i,
                    "feature": feature,
                    "ours": value,
                    "theirs": 1.0,
                }
            )
    return pd.DataFrame(out)


def test_moved_pairs_counts_changes_above_ten_percent() -> None:
    table = _pairs(
        [
            ("P", "burst_rate_per_min", 1.0, 1.05),
            ("P", "burst_rate_per_min", 1.0, 1.2),
            ("P", "burst_rate_per_min", 2.0, 1.0),
            ("Q", "burst_rate_per_min", 1.0, 1.0),
        ]
    )
    got = moved_pairs(table)
    assert abs(MOVE - 0.10) < 1e-12
    assert "post hoc" in got["label"].lower()
    assert got["by_plate"]["P"]["burst_rate_per_min"] == {"n": 3, "moved": 2, "share": 0.6667}
    assert got["by_plate"]["pooled"]["burst_rate_per_min"]["moved"] == 2


def _threshold_rows() -> pd.DataFrame:
    rows = [
        ("P", 0.01, False, "valid_below_cutoff"),
        ("P", 0.02, False, "valid_below_cutoff"),
        ("P", 0.3, False, "valid_above_cutoff"),
        ("P", None, True, "no_intraburst_peak"),
        ("Q", None, True, "no_void_minimum"),
        ("Q", 2.0, True, "threshold_above_ceiling"),
    ]
    return pd.DataFrame(rows, columns=["plate", "threshold_s", "fallback", "reason"])


def test_fallback_record_per_plate_and_pooled() -> None:
    got = fallback_record(_threshold_rows())
    assert got["P"]["n_active"] == 4 and got["P"]["n_fallback"] == 1
    assert got["P"]["share"] == pytest.approx(0.25)
    assert got["Q"]["share"] == pytest.approx(1.0)
    assert got["pooled"]["n_active"] == 6
    assert got["P"]["by_reason"]["valid_above_cutoff"] == 1


def test_threshold_distribution_uses_valid_thresholds_in_ms() -> None:
    got = threshold_distribution(_threshold_rows())
    assert got["P"]["n_valid"] == 3
    assert got["P"]["median_ms"] == pytest.approx(20.0)
    assert got["P"]["share_valid_above_cutoff"] == pytest.approx(1 / 3, abs=1e-4)
    assert got["Q"]["n_valid"] == 0 and got["Q"]["median_ms"] is None
    assert got["P"]["share_valid_below_1ms"] == pytest.approx(0.0)


def test_results_text_is_json_with_one_threshold_row_per_line() -> None:
    rows = _threshold_rows().assign(div=7, electrode="A1_11", n_spikes=900)
    text = results_text({"registration": {"file": "preregistration/extractor_logisi.json"}}, rows)
    doc = json.loads(text)
    assert doc["registration"]["file"] == "preregistration/extractor_logisi.json"
    assert doc["threshold_columns"][4] == "threshold_s"
    assert len(doc["thresholds"]) == 6
    assert doc["thresholds"][0] == ["P", 7, "A1_11", 900, 0.01, "valid_below_cutoff"]
    assert doc["thresholds"][3][4] is None


def test_s4_runs_through_the_shared_pipeline(tmp_path: Path) -> None:
    root = _write_plate(tmp_path)
    table = paired_table(("P1",), root=root, definitions=(*LADDER, LOGISI), processes=2)
    s4 = table[table["variant"] == "S4_logisi"]
    s3 = table[table["variant"] == "S3_rate_per_bursting_electrode"]
    assert len(s4) == len(s3) > 0
    rows = thresholds_table(("P1",), root=root, processes=2)
    # Above 10 spikes/min, noisy dropped per DIV: _11, _13, _14 in each well and DIV,
    # except A1_14 at DIV7.
    assert len(rows) == 3 * 4 - 1
    assert set(rows.columns) >= {"plate", "well", "div", "electrode", "threshold_s", "fallback"}


def test_the_logisi_plot_creates_its_directory(tmp_path: Path) -> None:
    stats = {
        v: {"pooled": {"burst_rate_per_min": {"median_ratio": 0.9}}, "P": {}} for v in VARIANTS
    }
    path = plot_logisi(stats, ("P",), tmp_path / "new" / "f.png")
    assert path.exists()
    assert np.isfinite(path.stat().st_size)

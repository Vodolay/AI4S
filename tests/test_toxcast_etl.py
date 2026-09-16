"""Tests for the ToxCast MEA selection/inventory step and the metrics writer."""

import json
from pathlib import Path

import pandas as pd

from neuromps.etl.toxcast_mea import endpoint_summary, select_mea
from neuromps.metrics import update_metrics


def _frame() -> pd.DataFrame:
    rows = [
        # aeid inside 2454-2541 but NOT an MEA assay: must be dropped
        (2484, "CCTE_Deisenroth_AIME_96WELL_LUC_Active", 1, "s1", 1.0, 50.0, 10.0, 0.03, 30.0),
        (2456, "CCTE_Shafer_MEA_acute_firing_rate_mean", 2, "s2", 1.0, 99.0, 70.97, 0.03, 30.0),
        (2456, "CCTE_Shafer_MEA_acute_firing_rate_mean", 3, "s3", 1.0, -169.0, 70.97, 0.03, 30.0),
        (2456, "CCTE_Shafer_MEA_acute_firing_rate_mean", 4, "s4", 0.1, 5.0, 70.97, 0.03, 30.0),
        (2494, "CCTE_Shafer_MEA_dev_firing_rate_mean", 2, "s5", 0.95, -20.0, 71.07, 0.3, 3000.0),
    ]
    cols = ["aeid", "aenm", "chid", "spid", "hitc", "top", "coff", "conc_min", "conc_max"]
    return pd.DataFrame(rows, columns=cols)


def test_select_mea_excludes_non_mea_aeids_in_range() -> None:
    out = select_mea(_frame())
    assert set(out["aeid"]) == {2456, 2494}
    assert "CCTE_Deisenroth_AIME_96WELL_LUC_Active" not in set(out["aenm"])


def test_select_mea_tags_assay_and_endpoint() -> None:
    out = select_mea(_frame()).set_index("spid")
    assert out.loc["s2", "assay"] == "acute"
    assert out.loc["s2", "endpoint"] == "firing_rate_mean"
    assert out.loc["s5", "assay"] == "dev"


def test_no_unverified_direction_column() -> None:
    """Direction is per-endpoint and unresolved (OQ-1): do not expose a signed column."""
    out = select_mea(_frame())
    assert "activity_change" not in out.columns
    assert out.set_index("spid").loc["s2", "top"] == 99.0  # raw value preserved


def test_endpoint_summary_counts_hits_by_direction() -> None:
    s = endpoint_summary(select_mea(_frame())).set_index("aeid")
    row = s.loc[2456]
    assert row["samples"] == 3
    assert row["hits"] == 2  # hitc 0.1 is not a hit
    assert row["hits_top_positive"] == 1
    assert row["hits_top_negative"] == 1
    assert row["chemicals"] == 3


def test_update_metrics_merges_sections(tmp_path: Path) -> None:
    p = tmp_path / "metrics.json"
    update_metrics("a", {"x": 1}, p)
    doc = update_metrics("b", {"y": 2}, p)
    assert doc == {"a": {"x": 1}, "b": {"y": 2}}
    assert json.loads(p.read_text())["a"]["x"] == 1

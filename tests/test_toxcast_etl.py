"""Tests for the ToxCast MEA selection/inventory step and the metrics writer."""

import json
from pathlib import Path

import pandas as pd
import pytest

from neuromps.etl.toxcast_mea import check_direction_convention, endpoint_summary, select_mea
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


def test_activity_change_sign_is_per_assay() -> None:
    """acute: +top means the quantity fell. dev: +top means it rose. OQ-1."""
    out = select_mea(_frame()).set_index("spid")
    assert out.loc["s2", "activity_change"] == -99.0  # acute, top=+99 -> fell
    assert out.loc["s3", "activity_change"] == 169.0  # acute, top=-169 -> rose
    assert out.loc["s5", "activity_change"] == -20.0  # dev, top=-20 -> fell


def test_viability_endpoints_get_no_signed_value() -> None:
    df = pd.DataFrame(
        [(2540, "CCTE_Shafer_MEA_acute_LDH", 9, "s9", 1.0, 50.0, 5.3, 0.03, 40.0)],
        columns=["aeid", "aenm", "chid", "spid", "hitc", "top", "coff", "conc_min", "conc_max"],
    )
    out = select_mea(df).iloc[0]
    assert out["is_viability"]
    assert pd.isna(out["activity_change"])


def _anchor_frame(acute_sign: int) -> pd.DataFrame:
    """One cytotoxic chemical: LDH hit plus firing-rate hits with the given raw sign."""
    rows = [(2540, "CCTE_Shafer_MEA_acute_LDH", 7, "v1", 1.0, 50.0, 5.3, 0.03, 40.0)]
    rows += [
        (
            2456,
            "CCTE_Shafer_MEA_acute_firing_rate_mean",
            7,
            f"a{i}",
            1.0,
            acute_sign * 99.0,
            70.97,
            0.03,
            40.0,
        )
        for i in range(10)
    ]
    cols = ["aeid", "aenm", "chid", "spid", "hitc", "top", "coff", "conc_min", "conc_max"]
    return select_mea(pd.DataFrame(rows, columns=cols))


def test_direction_check_passes_on_expected_convention() -> None:
    assert check_direction_convention(_anchor_frame(+1)) == {"acute": 1.0}


def test_direction_check_raises_when_convention_flips() -> None:
    with pytest.raises(ValueError, match="ACTIVITY_SIGN"):
        check_direction_convention(_anchor_frame(-1))


def test_endpoint_summary_counts_hits_by_direction() -> None:
    s = endpoint_summary(select_mea(_frame())).set_index("aeid")
    row = s.loc[2456]
    assert row["samples"] == 3
    assert row["hits"] == 2  # hitc 0.1 is not a hit
    assert row["hits_quantity_up"] == 1
    assert row["hits_quantity_down"] == 1
    assert row["chemicals"] == 3


def test_viability_endpoints_are_not_counted_by_direction() -> None:
    df = pd.DataFrame(
        [(2540, "CCTE_Shafer_MEA_acute_LDH", 9, "s9", 1.0, 50.0, 5.3, 0.03, 40.0)],
        columns=["aeid", "aenm", "chid", "spid", "hitc", "top", "coff", "conc_min", "conc_max"],
    )
    row = endpoint_summary(select_mea(df)).iloc[0]
    assert row["hits"] == 1
    assert row["hits_quantity_up"] == 0
    assert row["hits_quantity_down"] == 0


def test_update_metrics_merges_sections(tmp_path: Path) -> None:
    p = tmp_path / "metrics.json"
    update_metrics("a", {"x": 1}, p)
    doc = update_metrics("b", {"y": 2}, p)
    assert doc == {"a": {"x": 1}, "b": {"y": 2}}
    assert json.loads(p.read_text())["a"]["x"] == 1

"""Agreement between our extractor and the depositors' meaRtools output."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neuromps.models.extractor_agreement import (
    MEARTOOLS_FILES,
    RANK_AGREEMENT,
    SCALE_TOLERANCE,
    UNCHECKABLE,
    agreement_by_feature,
    plate_prefix,
    read_meartools,
    summarise,
)


def test_the_two_feature_lists_together_cover_the_nine() -> None:
    """Every feature is either checked against meaRtools or named as uncheckable."""
    from neuromps.features.mea import FEATURE_NAMES

    assert set(MEARTOOLS_FILES) | set(UNCHECKABLE) == set(FEATURE_NAMES)
    assert not set(MEARTOOLS_FILES) & set(UNCHECKABLE)


def test_plate_prefix_is_read_off_the_spike_files(tmp_path: Path) -> None:
    (tmp_path / "hPSC_20517_MEA1_DIV10_spikes.csv").write_text("Channel,Time\n")
    (tmp_path / "hPSC_20517_MEA1_DIV14_spikes.csv").write_text("Channel,Time\n")
    assert plate_prefix(tmp_path) == "hPSC_20517_MEA1"


def test_plate_prefix_is_none_when_there_is_nothing_to_read(tmp_path: Path) -> None:
    assert plate_prefix(tmp_path) is None
    (tmp_path / "noisy_electrodes_hPSC_MEA1.csv").write_text("x\n")
    assert plate_prefix(tmp_path) is None


def test_read_meartools_melts_the_div_columns(tmp_path: Path) -> None:
    (tmp_path / "P_bursts_per_min.csv").write_text(
        "well,treatment,divDIV7,divDIV21\nA3,NoTreatment,1.5,4.5\nA4,NoTreatment,2.0,\n"
    )
    out = read_meartools(tmp_path, "P")
    assert set(out.columns) == {"well", "div", "feature", "theirs"}
    assert set(out["feature"]) == {"burst_rate_per_min"}
    assert sorted(out["div"].unique()) == [7, 21]
    assert float(out[(out["well"] == "A3") & (out["div"] == 21)]["theirs"].iloc[0]) == 4.5
    # A missing cell survives as NaN rather than shifting the table.
    assert out["theirs"].isna().sum() == 1


def test_read_meartools_skips_a_feature_whose_file_is_absent(tmp_path: Path) -> None:
    (tmp_path / "P_STTC.csv").write_text("well,divDIV7\nA3,0.5\n")
    out = read_meartools(tmp_path, "P")
    assert set(out["feature"]) == {"synchrony_index"}


def _paired(feature: str, ours: list[float], theirs: list[float], wells=None) -> pd.DataFrame:
    wells = wells or [f"W{i}" for i in range(len(ours))]
    return pd.DataFrame(
        {
            "plate": "P",
            "well": wells,
            "div": list(range(len(ours))),
            "feature": feature,
            "ours": ours,
            "theirs": theirs,
        }
    )


def test_a_feature_that_matches_exactly_reproduces() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    out = agreement_by_feature(_paired("burst_duration_s", values, values))["burst_duration_s"]
    assert out["spearman_rho"] == 1.0
    assert out["median_ratio_ours_over_theirs"] == 1.0
    assert out["fraction_within_10pct"] == 1.0
    assert out["reproduces"] is True


def test_a_feature_on_a_different_scale_orders_the_same_but_does_not_reproduce() -> None:
    theirs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    ours = [0.5 * v for v in theirs]
    out = agreement_by_feature(_paired("burst_rate_per_min", ours, theirs))["burst_rate_per_min"]
    assert out["spearman_rho"] == 1.0
    assert out["median_ratio_ours_over_theirs"] == 0.5
    assert out["reproduces"] is False
    assert summarise({"burst_rate_per_min": out})["same_ordering_different_scale"] == [
        "burst_rate_per_min"
    ]


def test_a_feature_that_orders_differently_is_separated_from_a_rescaling() -> None:
    theirs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    ours = [3.0, 1.0, 6.0, 2.0, 5.0, 4.0]
    out = agreement_by_feature(_paired("synchrony_index", ours, theirs))["synchrony_index"]
    assert out["spearman_rho"] < RANK_AGREEMENT
    assert summarise({"synchrony_index": out})["disagree_on_ordering_too"] == ["synchrony_index"]


def test_scale_tolerance_is_applied_symmetrically() -> None:
    theirs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    inside = [v * (1 + SCALE_TOLERANCE / 2) for v in theirs]
    outside = [v * (1 + SCALE_TOLERANCE * 2) for v in theirs]
    assert agreement_by_feature(_paired("mfr_hz", inside, theirs))["mfr_hz"]["reproduces"] is True
    assert agreement_by_feature(_paired("mfr_hz", outside, theirs))["mfr_hz"]["reproduces"] is False


def test_the_bootstrap_clusters_on_wells_not_rows() -> None:
    """One well contributing many DIVs must not count as many independent points."""
    theirs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    ours = [v + 0.01 for v in theirs]
    one_well = agreement_by_feature(_paired("mfr_hz", ours, theirs, wells=["A1"] * 8))["mfr_hz"]
    many = agreement_by_feature(_paired("mfr_hz", ours, theirs, wells=[f"W{i}" for i in range(8)]))[
        "mfr_hz"
    ]
    assert one_well["n_wells"] == 1
    assert many["n_wells"] == 8
    # With every point from one well the interval cannot be narrower.
    assert one_well["spearman_rho_ci95"][0] <= many["spearman_rho_ci95"][0]


def test_too_few_points_is_reported_rather_than_scored() -> None:
    out = agreement_by_feature(_paired("mfr_hz", [1.0, 2.0], [1.0, 2.0]))["mfr_hz"]
    assert "note" in out and "spearman_rho" not in out


def test_a_zero_denominator_does_not_poison_the_ratio() -> None:
    theirs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    ours = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    out = agreement_by_feature(_paired("nae_like", ours, theirs))["nae_like"]
    assert np.isfinite(out["median_ratio_ours_over_theirs"])


@pytest.mark.parametrize("feature", sorted(MEARTOOLS_FILES))
def test_every_mapped_feature_names_a_plausible_file_stem(feature: str) -> None:
    stem = MEARTOOLS_FILES[feature]
    assert stem and "/" not in stem and not stem.endswith(".csv")

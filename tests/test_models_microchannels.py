"""Tests for the microchannel directionality analysis on the Mateus deposit."""

import numpy as np
import pandas as pd

from neuromps.models.microchannels import (
    chip_identity,
    condition_comparison,
    count_confound,
    per_chip,
)


def _recordings() -> pd.DataFrame:
    rows = []
    for condition, chips, asym in (("control", 4, 0.3), ("rams", 4, 0.8)):
        for chip in range(chips):
            for div in (12, 17):
                rows.append(
                    {
                        "experiment": f"EXP{chip}",
                        "chip": f"chip_{chip}",
                        "condition": condition,
                        "div": div,
                        "asymmetry": asym + 0.01 * div * (1 if chip % 2 else -1),
                        "forward_events": 500,
                        "backward_events": 100,
                    }
                )
    return pd.DataFrame(rows)


def test_chip_identity_separates_chips_across_experiments() -> None:
    assert chip_identity("EXP1", "chip_2") == "EXP1/chip_2"
    assert chip_identity("EXP2", "chip_2") != chip_identity("EXP1", "chip_2")


def test_per_chip_collapses_repeated_recordings() -> None:
    """Three DIVs of one chip are one chip, not three independent measurements."""
    chips = per_chip(_recordings())
    assert len(chips) == 8
    assert set(chips["condition"]) == {"control", "rams"}
    assert chips["n_recordings"].eq(2).all()


def test_condition_comparison_reports_n_chips_not_n_recordings() -> None:
    result = condition_comparison(per_chip(_recordings()), reference="control")
    assert result["n_chips"] == 8
    assert result["by_condition"]["rams"]["n_chips"] == 4
    assert "kruskal_p" in result


def test_a_real_difference_is_detected() -> None:
    result = condition_comparison(per_chip(_recordings()), reference="control")
    assert (
        result["by_condition"]["rams"]["median_abs_asymmetry"]
        > (result["by_condition"]["control"]["median_abs_asymmetry"])
    )


def test_identical_conditions_are_not_called_different() -> None:
    frame = _recordings()
    frame["asymmetry"] = 0.5
    result = condition_comparison(per_chip(frame), reference="control")
    assert result["kruskal_p"] is None
    assert "undefined" in result["note"]


def test_count_confound_detects_asymmetry_riding_on_event_count() -> None:
    """Few events make |asymmetry| large by chance; that must be visible, not hidden."""
    rng = np.random.default_rng(0)
    chips = pd.DataFrame(
        {
            "chip": [f"c{i}" for i in range(40)],
            "condition": ["control"] * 20 + ["rams"] * 20,
            "total_events": np.concatenate([rng.integers(20, 60, 20), rng.integers(500, 900, 20)]),
        }
    )
    # |asymmetry| falls as events rise: the signature of a counting artefact.
    chips["abs_asymmetry"] = 12.0 / np.sqrt(chips["total_events"])
    out = count_confound(chips)
    assert out["spearman_rho"] < -0.8
    assert out["p_value"] < 0.01


def test_count_confound_is_quiet_when_there_is_none() -> None:
    rng = np.random.default_rng(1)
    chips = pd.DataFrame(
        {
            "chip": [f"c{i}" for i in range(40)],
            "condition": ["control"] * 20 + ["rams"] * 20,
            "total_events": rng.integers(100, 900, 40),
            "abs_asymmetry": rng.uniform(0.2, 0.9, 40),
        }
    )
    assert count_confound(chips)["p_value"] > 0.05


def test_a_chip_without_detected_channels_is_excluded_not_zeroed() -> None:
    frame = _recordings()
    frame.loc[0, "asymmetry"] = np.nan
    chips = per_chip(frame)
    assert chips["n_recordings"].min() >= 1
    assert chips["abs_asymmetry"].notna().all()


def test_comparison_needs_two_conditions() -> None:
    frame = _recordings()
    frame = frame[frame["condition"] == "control"]
    result = condition_comparison(per_chip(frame), reference="control")
    assert result["kruskal_p"] is None
    assert "too few" in result["note"]

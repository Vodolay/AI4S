"""Cutting the EPA sessions into records, which the dose logfile alone makes possible."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neuromps.etl import vassallo


def _logfile_frame() -> pd.DataFrame:
    """Two header rows then one experiment, shaped like Logfile_EPAv1.xls."""
    row = [None] * 31
    row[0], row[1] = "Domoic Acid ", "102010 15160 10309 DA -- 10309.txt"
    row[4], row[5] = 300, 3300  # baseline window
    row[6], row[7], row[8] = 0.01, 3900, 5700  # conc1
    row[9], row[10], row[11] = 0.1, 6300, 8100  # conc2
    return pd.DataFrame([[None] * 31, [None] * 31, row])


def test_the_baseline_and_every_concentration_become_windows() -> None:
    entries = vassallo.dose_windows(_logfile_frame())
    assert len(entries) == 1
    windows = entries[0]["windows"]
    assert [w["label"] for w in windows] == ["baseline", "conc1", "conc2"]
    assert windows[0]["dose_um"] is None
    assert [w["dose_um"] for w in windows[1:]] == [0.01, 0.1]
    assert windows[2]["start_s"] == 6300 and windows[2]["end_s"] == 8100


def test_a_row_with_missing_dose_columns_stops_cleanly() -> None:
    """Chips differ in how many concentrations they received."""
    frame = _logfile_frame()
    frame.iloc[2, 9] = None  # conc2 dose absent
    windows = vassallo.dose_windows(frame)[0]["windows"]
    assert [w["label"] for w in windows] == ["baseline", "conc1"]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("102010 15160 10309 DA -- 10309.txt", ("102010", "10309")),
        ("111610 15222 15154 13105 ACE DP GLY -- 15154.txt", ("111610", "15154")),
        ("110310 10307 11821 COX--10307.txt", ("110310", "10307")),
    ],
)
def test_the_join_key_survives_both_namings(name: str, expected: tuple[str, str]) -> None:
    """A logfile row names every compound on the plate; the file names one."""
    assert vassallo.session_and_chip(name) == expected


def test_spikes_are_cut_to_the_window_and_timed_from_its_start() -> None:
    fs = vassallo.SAMPLE_RATE_HZ
    ticks = (np.array([100.0, 310.0, 320.0, 700.0, 3400.0]) * fs).astype(np.int64)
    table = pd.DataFrame({"tick": ticks, "channel": [12, 12, 13, 12, 12]})
    spikes = vassallo.window_spikes(table, 300.0, 3300.0)
    assert set(spikes) == {"12", "13"}
    # 310 s and 700 s fall inside; they are reported 10 s and 400 s after the start.
    assert spikes["12"] == pytest.approx([10.0, 400.0])
    assert spikes["13"] == pytest.approx([20.0])


def test_spikes_between_windows_are_dropped_not_reassigned() -> None:
    """The minutes while a compound is pipetted belong to no concentration."""
    fs = vassallo.SAMPLE_RATE_HZ
    table = pd.DataFrame(
        {"tick": (np.array([3500.0, 3800.0]) * fs).astype(np.int64), "channel": [12, 12]}
    )
    assert vassallo.window_spikes(table, 300.0, 3300.0) == {}
    assert vassallo.window_spikes(table, 3900.0, 5700.0) == {}


def test_a_dose_carries_its_mechanism_and_a_baseline_carries_none() -> None:
    treated = vassallo._compound("domoic acid", {"dose_um": 0.1})
    assert treated["role"] == "test"
    assert treated["mechanism_class"] == "glutamatergic_agonist"
    assert (treated["dose_value"], treated["dose_unit"]) == (0.1, "uM")
    baseline = vassallo._compound("domoic acid", {"dose_um": None})
    assert baseline["role"] == "untreated_control"
    assert baseline["mechanism_class"] is None


def test_every_compound_in_the_study_has_a_mechanism() -> None:
    """A missing entry would silently become a null label in the classifier."""
    expected = {
        "deltamethrin",
        "domoic acid",
        "chlorpyrifos-oxon",
        "acetaminophen",
        "glyphosate",
        "dimethyl phthalate",
    }
    assert set(vassallo.MECHANISM) == expected
    assert all(vassallo.MECHANISM.values())

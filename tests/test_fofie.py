"""Parsing the Fofie workbook, and the rate-only arm built on it."""

import numpy as np
import pandas as pd
import pytest

from neuromps.etl.fofie import (
    WORKBOOK,
    WellSeries,
    deduplicate,
    exposure_timepoint,
    is_usable,
    mechanism_of,
    normalise_condition,
    parse_blocks,
    parse_row_per_well,
)


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("GABA [10 µM]", ("gaba", 10.0, "uM")),
        ("TTX (5 nM)", ("ttx", 5.0, "nM")),
        ("DNQX (100 µM)", ("dnqx", 100.0, "uM")),
        ("DMSO (0.25%)", ("dmso", None, None)),
        # The whole matched phrase, not its first word: see normalise_condition.
        ("No Intervention", ("no intervention", None, None)),
        ("Conotoxin [0.1 µM]", ("conotoxin", 0.1, "uM")),
    ],
)
def test_condition_labels_split_into_compound_dose_and_unit(label, expected) -> None:
    assert normalise_condition(label) == expected


def test_a_label_without_a_unit_yields_no_dose() -> None:
    """`TTX 10` is a number, not a dose; the sheet holding those is skipped."""
    compound, dose, unit = normalise_condition("TTX 10")
    assert dose is None and unit is None


def test_mechanism_map_covers_the_compounds_and_calls_vehicles_controls() -> None:
    assert mechanism_of("ttx") == "na_channel_blocker"
    assert mechanism_of("dnqx") == "glutamatergic_antagonist"
    assert mechanism_of("gaba") == "gabaa_agonist_or_pam"
    assert mechanism_of("dmso") == "control"
    assert mechanism_of("no intervention") == "control"
    assert mechanism_of("something-unheard-of") is None
    # A bare "no" is not a vehicle name, and must not be read as one.
    assert mechanism_of("no") is None


def test_the_control_label_matches_the_rest_of_the_asset() -> None:
    """A different word here would hide this source's controls from every
    cross-source comparison, which is the one thing they exist for."""
    assert mechanism_of("dpbs") == "control"


def _series(condition: str, counts: dict) -> WellSeries:
    return WellSeries("sheet", "block", "A1", condition, counts)


def test_baseline_is_found_whatever_the_other_timepoints_are() -> None:
    s = _series("TTX (5 nM)", {"Base": 100.0, "1hr": 50.0, "24hrs": 60.0})
    assert s.baseline == 100.0
    assert _series("TTX (5 nM)", {"1hr": 50.0}).baseline is None


def test_exposure_prefers_two_hours_and_never_a_washout() -> None:
    assert exposure_timepoint(_series("x", {"Base": 1, "1hr": 2, "2hrs": 3})) == "2hrs"
    assert exposure_timepoint(_series("x", {"Base": 1, "1hr": 2})) == "1hr"
    # A post-washout reading measures recovery, so it is not an exposure.
    assert exposure_timepoint(_series("x", {"Base": 1, "1hr-PW": 2})) is None
    # Nor is a temperature repeat, which re-measures a well already counted.
    assert exposure_timepoint(_series("x", {"Base": 1, "2hrs+T": 2})) is None


def test_usable_requires_a_known_compound_a_baseline_and_two_points() -> None:
    assert is_usable(_series("TTX (5 nM)", {"Base": 100.0, "2hrs": 40.0}))
    assert not is_usable(_series("A8", {"Base": 100.0, "2hrs": 40.0}))  # a well id
    assert not is_usable(_series("G", {"Base": 100.0, "2hrs": 40.0}))  # a row letter
    assert not is_usable(_series("TTX (5 nM)", {"Base": 0.0, "2hrs": 40.0}))
    assert not is_usable(_series("TTX (5 nM)", {"Base": 100.0}))


def test_deduplicate_drops_a_block_read_twice_but_keeps_a_replicate() -> None:
    same = [_series("TTX (5 nM)", {"Base": 100.0, "2hrs": 40.0})] * 2
    assert len(deduplicate(same)) == 1
    different = [
        _series("TTX (5 nM)", {"Base": 100.0, "2hrs": 40.0}),
        _series("TTX (5 nM)", {"Base": 110.0, "2hrs": 44.0}),
    ]
    assert len(deduplicate(different)) == 2


def test_row_per_well_layout_is_parsed() -> None:
    frame = pd.DataFrame(
        [
            [None, "Base", "2hrs"],
            ["DPBS", 1000, 990],
            ["DPBS", 1100, 1080],
            ["GABA [50 µM]", 1200, 400],
            ["GABA [50 µM]", 1300, 420],
        ]
    )
    found = [s for s in parse_row_per_well(frame, "sheet") if is_usable(s)]
    assert len(found) == 4
    assert {s.condition for s in found} == {"DPBS", "GABA [50 µM]"}
    gaba = [s for s in found if s.condition.startswith("GABA")][0]
    assert gaba.counts == {"Base": 1200.0, "2hrs": 400.0}


def test_block_layout_is_parsed() -> None:
    frame = pd.DataFrame(
        [
            ["TTX (5 nM)", None, None, None],
            [None, None, "A8", "B8"],
            [None, "Base", 900, 800],
            [None, "2hrs", 300, 260],
        ]
    )
    found = [s for s in parse_blocks(frame, "sheet") if is_usable(s)]
    assert len(found) == 2
    assert {s.well for s in found} == {"A8", "B8"}
    assert found[0].counts["Base"] == 900.0


def test_a_percentage_block_is_not_mistaken_for_counts() -> None:
    """The sheets put derived %-change beside the counts; only counts are taken."""
    frame = pd.DataFrame(
        [
            [None, "Base", "2hrs"],
            ["GABA [50 µM]", 0, -35.7],
            ["GABA [50 µM]", 0, -41.2],
        ]
    )
    assert [s for s in parse_row_per_well(frame, "sheet") if is_usable(s)] == []


@pytest.mark.skipif(not WORKBOOK.exists(), reason="fofie2025 not fetched")
def test_the_real_workbook_parses_with_controls_flat_and_blockers_not() -> None:
    """The check that the parse is right, rather than merely non-empty.

    If columns or rows were misaligned, untreated wells would not sit near zero
    while sodium-channel blockers sit far below it.
    """
    from neuromps.models.rate_only import direction_agreement, fofie_pairs

    pairs = fofie_pairs()
    assert len(pairs) > 100
    assert pairs["group"].nunique() >= 8
    # No block may hold two compounds, or the grouping would leak a label.
    assert (pairs.groupby("group")["compound"].nunique() > 1).sum() == 0
    agreement = direction_agreement(pairs)
    assert agreement["control_is_flat_and_blocker_is_not"] is True
    assert abs(agreement["control"]["median_log2_change"]) < 0.5
    assert agreement["na_channel_blocker"]["median_log2_change"] < -1.0


@pytest.mark.skipif(not WORKBOOK.exists(), reason="fofie2025 not fetched")
def test_the_rate_delta_is_a_ratio_so_duration_cancels() -> None:
    """The property that lets a source without a recording duration be used."""
    from neuromps.etl.fofie import build_pairs

    pairs = build_pairs()
    row = pairs.iloc[0]
    expected = np.log2(row["exposure_count"] / row["baseline_count"])
    assert row["delta_log2_rate"] == pytest.approx(expected)

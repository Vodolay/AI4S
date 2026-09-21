"""Tests for reading an asset that may have been built without every source."""

import pandas as pd

from neuromps.asset import COMPARTMENT_COLUMN, compartment_records


def _single_chamber() -> pd.DataFrame:
    """What a --quick run builds: no source contributes a device block."""
    return pd.DataFrame([{"source": "kapucu2022", "unit_id": "w1", "features.mfr_hz": 1.0}])


def test_missing_column_gives_an_empty_frame_not_a_keyerror() -> None:
    out = compartment_records(_single_chamber())
    assert out.empty


def test_empty_result_still_carries_the_compartment_column() -> None:
    """Callers filter the result further; the column has to be there to filter on."""
    out = compartment_records(_single_chamber())
    assert COMPARTMENT_COLUMN in out.columns
    assert out[out[COMPARTMENT_COLUMN].isin(["red", "green"])].empty


def test_does_not_add_the_column_to_the_caller_s_frame() -> None:
    records = _single_chamber()
    compartment_records(records)
    assert COMPARTMENT_COLUMN not in records.columns


def test_keeps_compartment_rows_and_drops_null_ones() -> None:
    records = pd.DataFrame(
        [
            {"unit_id": "a", COMPARTMENT_COLUMN: "red"},
            {"unit_id": "b", COMPARTMENT_COLUMN: "green"},
            {"unit_id": "c", COMPARTMENT_COLUMN: None},
        ]
    )
    out = compartment_records(records)
    assert list(out["unit_id"]) == ["a", "b"]

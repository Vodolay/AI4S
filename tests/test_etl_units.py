"""Tests for dose-unit normalization shared by the ETL modules."""

import pytest

from neuromps.etl.units import SCHEMA_DOSE_UNITS, normalize_dose


def test_micromolar_and_nanomolar_pass_through() -> None:
    assert normalize_dose(30.0, "uM") == (30.0, "uM")
    assert normalize_dose(100.0, "nM") == (100.0, "nM")


def test_millimolar_is_converted_not_relabelled() -> None:
    """The schema has no mM, so a mM dose must be rescaled, not renamed."""
    assert normalize_dose(5.0, "mM") == (5000.0, "uM")
    assert normalize_dose(0.5, "mM") == (500.0, "uM")


def test_units_are_case_insensitive_and_trimmed() -> None:
    assert normalize_dose(1.0, " Um ") == (1.0, "uM")
    assert normalize_dose(1.0, "NM") == (1.0, "nM")
    assert normalize_dose(2.0, "MM") == (2000.0, "uM")


def test_unknown_unit_is_rejected_rather_than_guessed() -> None:
    with pytest.raises(ValueError, match="unknown dose unit"):
        normalize_dose(1.0, "molar")


def test_every_output_unit_is_one_the_schema_allows() -> None:
    for unit in ("uM", "nM", "mM"):
        _, normalized = normalize_dose(1.0, unit)
        assert normalized in SCHEMA_DOSE_UNITS


def test_none_value_passes_through_untouched() -> None:
    assert normalize_dose(None, "uM") == (None, "uM")

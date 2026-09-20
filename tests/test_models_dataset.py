"""Tests for pairing exposure recordings with their baselines and forming deltas."""

import numpy as np
import pandas as pd
import pytest

from neuromps.models.dataset import (
    DELTA_FEATURES,
    build_pairs,
    delta_columns,
    log_ratio,
)


def _row(**kw) -> dict:
    base = {
        "record_id": "r",
        "source": "kapucu2022",
        "data_level": "spike_times",
        "unit_id": "A8",
        "plate_id": "Rat_MEA2Pharma_DIV22",
        "cell_source.species": "rat",
        "cell_source.system": "primary_cortical",
        "compound.name": "gabazine",
        "compound.mechanism_class": "gabaa_antagonist",
        "compound.dose_value": 30.0,
        "compound.dose_unit": "uM",
        "recording.phase": "exposure",
        "qc.usable": True,
        "device.device_id": None,
        "device.compartment_id": None,
    }
    base.update({f"features.{f}": 1.0 for f in DELTA_FEATURES})
    base.update(kw)
    return base


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_log_ratio_is_zero_for_no_change_and_one_for_doubling() -> None:
    assert log_ratio(4.0, 4.0) == pytest.approx(0.0)
    assert log_ratio(8.0, 4.0) == pytest.approx(1.0)
    assert log_ratio(2.0, 4.0) == pytest.approx(-1.0)


def test_log_ratio_floors_silence_instead_of_diverging() -> None:
    assert np.isfinite(log_ratio(0.0, 4.0))
    assert log_ratio(0.0, 4.0) < -2  # "abolished", finite and comparable


def test_log_ratio_is_undefined_without_a_baseline() -> None:
    """Silent before, active during: the ratio does not exist, so do not invent one."""
    assert np.isnan(log_ratio(4.0, 0.0))
    assert np.isnan(log_ratio(None, 4.0))
    assert np.isnan(log_ratio(4.0, None))


def test_pairs_match_exposure_to_its_own_baseline() -> None:
    rows = [
        _row(record_id="exp", **{"features.mfr_hz": 8.0}),
        _row(
            record_id="base",
            plate_id="Rat_MEA2Baseline_DIV22",
            **{
                "recording.phase": "baseline",
                "compound.mechanism_class": None,
                "compound.name": "none",
                "features.mfr_hz": 4.0,
            },
        ),
    ]
    pairs = build_pairs(_frame(rows))
    assert len(pairs) == 1
    assert pairs.iloc[0]["label"] == "gabaa_antagonist"
    assert pairs.iloc[0]["delta.mfr_hz"] == pytest.approx(1.0)
    assert pairs.iloc[0]["group"] == "kapucu2022:rat:A8"


def test_exposure_without_a_baseline_is_dropped() -> None:
    pairs = build_pairs(_frame([_row(record_id="exp")]))
    assert pairs.empty


def test_control_exposures_are_labelled_control() -> None:
    rows = [
        _row(
            record_id="exp",
            **{
                "compound.name": "none",
                "compound.mechanism_class": None,
                "compound.dose_value": None,
            },
        ),
        _row(
            record_id="base",
            plate_id="Rat_MEA2Baseline_DIV22",
            **{
                "recording.phase": "baseline",
                "compound.mechanism_class": None,
                "compound.name": "none",
            },
        ),
    ]
    pairs = build_pairs(_frame(rows))
    assert pairs.iloc[0]["label"] == "control"


def test_unusable_records_are_excluded_by_default() -> None:
    rows = [
        _row(record_id="exp", **{"qc.usable": False}),
        _row(
            record_id="base", plate_id="Rat_MEA2Baseline_DIV22", **{"recording.phase": "baseline"}
        ),
    ]
    assert build_pairs(_frame(rows)).empty
    assert len(build_pairs(_frame(rows), usable_only=False)) == 1


def test_groups_never_span_train_and_test_by_construction() -> None:
    """One group per culture unit, so a well cannot appear on both sides of a split."""
    rows = []
    for well in ("A8", "B8"):
        rows += [
            _row(record_id=f"exp{well}", unit_id=well),
            _row(
                record_id=f"base{well}",
                unit_id=well,
                plate_id="Rat_MEA2Baseline_DIV22",
                **{"recording.phase": "baseline"},
            ),
        ]
    pairs = build_pairs(_frame(rows))
    assert set(pairs["group"]) == {"kapucu2022:rat:A8", "kapucu2022:rat:B8"}
    assert pairs["group"].nunique() == len(pairs)


def test_screenneuropharm_doses_pair_with_the_array_basal_recording() -> None:
    rows = [
        _row(
            record_id="d1",
            source="screenneuropharm_bic",
            unit_id="Br11-Cx-18331",
            plate_id="18331",
            **{"compound.name": "bicuculline", "compound.dose_value": 10.0, "features.mfr_hz": 2.0},
        ),
        _row(
            record_id="d2",
            source="screenneuropharm_bic",
            unit_id="Br11-Cx-18331",
            plate_id="18331",
            **{"compound.name": "bicuculline", "compound.dose_value": 30.0, "features.mfr_hz": 1.0},
        ),
        _row(
            record_id="b",
            source="screenneuropharm_bic",
            unit_id="Br11-Cx-18331",
            plate_id="18331",
            **{
                "recording.phase": "baseline",
                "compound.name": "none",
                "compound.mechanism_class": None,
                "compound.dose_value": None,
                "features.mfr_hz": 4.0,
            },
        ),
    ]
    pairs = build_pairs(_frame(rows)).sort_values("dose_value")
    assert list(pairs["delta.mfr_hz"].round(2)) == [-1.0, -2.0]
    assert pairs["group"].nunique() == 1  # same array: one group


def test_delta_columns_cover_every_feature() -> None:
    cols = delta_columns()
    assert len(cols) == len(DELTA_FEATURES)
    assert all(c.startswith("delta.") for c in cols)

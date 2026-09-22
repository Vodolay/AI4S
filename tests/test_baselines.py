"""The untreated-baseline comparison, and the culture identity it rests on."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neuromps.models import baselines


def _record(**over) -> dict:
    base = {
        "data_level": "spike_times",
        "qc.usable": True,
        "source": "kapucu2022",
        "plate_id": "Rat_MEA2Baseline_DIV22",
        "unit_id": "A1",
        "cell_source.species": "rat",
        "cell_source.system": "primary_cortical",
        "platform.vendor": "Axion Biosystems",
        "platform.electrodes_per_unit": 16,
        "compound.role": "untreated_control",
        "recording.phase": "baseline",
    }
    base.update({f"features.{f}": 1.0 for f in baselines.FEATURES})
    base.update(over)
    return base


def test_a_well_name_alone_would_merge_two_species() -> None:
    """Well 'A1' exists on the rat plate and on the human one."""
    rat = _record()
    human = _record(plate_id="hPSC_MEA3Baseline_DIV29", **{"cell_source.species": "human"})
    assert baselines.culture_id(pd.Series(rat)) != baselines.culture_id(pd.Series(human))


def test_a_longitudinal_source_keeps_one_identity_across_sessions() -> None:
    """The same organoid recorded weekly is one culture, not twenty."""
    first = pd.Series(_record(source="trujillo2019", plate_id="161217", unit_id="well04"))
    later = pd.Series(_record(source="trujillo2019", plate_id="170303", unit_id="well04"))
    assert baselines.culture_id(first) == baselines.culture_id(later) == "trujillo2019:well04"


def test_a_culture_spanning_two_species_is_refused() -> None:
    """The invariant that caught the identity bug in the first place."""
    rows = [
        _record(source="x", plate_id="p", unit_id="A1"),
        _record(source="x", plate_id="p", unit_id="A1", **{"cell_source.species": "human"}),
    ]
    with pytest.raises(ValueError, match="spanning more than one species"):
        baselines.baseline_records(pd.DataFrame(rows))


def test_only_untreated_usable_spike_time_records_are_used() -> None:
    rows = [
        _record(unit_id="A1"),
        _record(unit_id="A2", **{"compound.role": "test"}),
        _record(unit_id="A3", **{"qc.usable": False}),
        _record(unit_id="A4", data_level="fitted_curve"),
    ]
    kept = baselines.baseline_records(pd.DataFrame(rows))
    assert list(kept["unit_id"]) == ["A1"]


def test_culture_counts_add_up_to_the_total() -> None:
    """Human plus rat cultures must equal all cultures, or the grouping is wrong."""
    rng = np.random.default_rng(0)
    rows = []
    for species, plate in (("rat", "ratplate"), ("human", "humanplate")):
        for well in range(6):
            rows.append(
                _record(
                    plate_id=plate,
                    unit_id=f"A{well}",
                    **{"cell_source.species": species},
                    **{f"features.{f}": float(rng.random()) for f in baselines.FEATURES},
                )
            )
    result = baselines.compare(baselines.baseline_records(pd.DataFrame(rows)), "test")
    assert result["n_human_cultures"] + result["n_rat_cultures"] == result["n_cultures"]


def test_a_slice_with_one_culture_a_side_reports_no_intervals() -> None:
    rows = [
        _record(unit_id="A1"),
        _record(unit_id="B1", plate_id="h", **{"cell_source.species": "human"}),
    ]
    result = baselines.compare(baselines.baseline_records(pd.DataFrame(rows)), "thin")
    assert "too few cultures" in result["note"]
    assert result["features"] == {}


def test_an_interval_spanning_zero_is_not_called_a_separation() -> None:
    assert not baselines._excludes_zero({"ci95": [-0.5, 0.5]})
    assert baselines._excludes_zero({"ci95": [0.2, 0.9]})
    assert baselines._excludes_zero({"ci95": [-0.9, -0.2]})
    assert not baselines._excludes_zero({"ci95": [float("nan"), 0.5]})


def test_every_feature_is_classified_as_rate_like_or_not() -> None:
    """The portability claim is read off this split, so it must cover the vector."""
    assert set(baselines.RATE_LIKE) <= set(baselines.FEATURES)


def test_a_thin_slice_reports_no_verdict_rather_than_agreement() -> None:
    """A blocked source empties the slice; two absent lists must not read as agreement.

    This is the defect that broke CI: `compare` returns early without a feature
    list when it cannot resample, and the sensitivity check indexed that key.
    """
    rows = [_record(unit_id="A1"), _record(unit_id="B1", plate_id="h",
                                           **{"cell_source.species": "human"})]
    result = baselines.run(pd.DataFrame(rows))
    assert result["headline_survives_dropping_exposure_phase_controls"] is None

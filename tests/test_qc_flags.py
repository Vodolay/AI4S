"""Tests for cross-record QC: replicate divergence, outliers and missing baseline pairs."""

import copy

import pytest

from neuromps.qc.flags import DEFAULT_Z, add_cross_record_flags, replicate_group, robust_z
from neuromps.qc.schema import validate_record


def _record(unit: str, mfr: float, phase: str = "exposure", compound: str = "gabazine") -> dict:
    return {
        "record_id": f"kapucu2022:plate:{unit.lower()}:{phase}",
        "source": "kapucu2022",
        "license": "CC-BY-4.0",
        "citation": "Kapucu FE et al. Sci Data 9, 120 (2022). doi:10.1038/s41597-022-01242-4",
        "data_level": "spike_times",
        "cell_source": {"species": "rat", "system": "primary_cortical"},
        "platform": {
            "vendor": "Axion Biosystems",
            "system": "Maestro 768-channel",
            "plate_format": "48-well",
            "electrodes_per_unit": 16,
            "sampling_rate_hz": 12500.0,
        },
        "plate_id": f"Rat_MEA2{'Baseline' if phase == 'baseline' else 'Pharma'}_DIV22",
        "unit_id": unit,
        "culture_age_days": 22,
        "compound": {
            "name": compound,
            "role": "test" if compound != "none" else "untreated_control",
            "mechanism_class": "gabaa_antagonist" if compound == "gabazine" else None,
            "dose_value": 30.0 if compound != "none" else None,
            "dose_unit": "uM" if compound != "none" else None,
        },
        "recording": {"duration_s": 1800.0, "phase": phase},
        "features": {"mfr_hz": mfr, "burst_rate_per_min": 5.0, "active_electrodes": 16},
        "feature_method": {
            "computed_by": "neuromps.features.mea",
            "spike_detection": "publisher: authors' MATLAB detection",
            "burst_method": "maxinterval_simple(isi<=0.1s,>=5spikes)",
        },
        "qc": {"usable": True, "flags": [], "n_noisy_electrodes": 0},
        "provenance": {
            "source_files": [{"path": "kapucu2022/x.csv", "sha256": "a" * 64}],
            "transform": ["read"],
            "generated_by": "test",
            "generated_at": "2026-09-16T00:00:00+00:00",
        },
    }


def test_robust_z_is_zero_at_the_median_and_scales_by_spread() -> None:
    z = robust_z([8.0, 9.0, 10.0, 11.0, 12.0, 40.0])
    assert z[2] < 1
    assert z[-1] > DEFAULT_Z


def test_robust_z_handles_zero_spread_without_dividing_by_zero() -> None:
    z = robust_z([5.0, 5.0, 5.0])
    assert all(v == 0.0 for v in z)


def test_robust_z_still_sees_an_outlier_when_the_mad_is_zero() -> None:
    """Replicates often share a value exactly, making the MAD zero; the outlier must survive."""
    z = robust_z([10.0, 10.0, 10.0, 10.0, 40.0])
    assert z[0] == pytest.approx(0.0)
    assert z[-1] > DEFAULT_Z


def test_replicate_group_keys_on_condition_not_on_well() -> None:
    a, b = _record("A8", 1.0), _record("B8", 2.0)
    assert replicate_group(a) == replicate_group(b)
    assert replicate_group(_record("A8", 1.0, compound="cnqx")) != replicate_group(a)


def test_divergent_replicate_is_flagged_and_its_peers_are_not() -> None:
    records = [_record(f"{r}8", 2.0) for r in "ABCD"] + [_record("E8", 50.0)]
    out = {r["unit_id"]: r for r in add_cross_record_flags(copy.deepcopy(records))}
    assert "replicate_divergence" in out["E8"]["qc"]["flags"]
    assert all("replicate_divergence" not in out[f"{r}8"]["qc"]["flags"] for r in "ABCD")


def test_divergence_note_names_the_feature_and_score() -> None:
    """A well can sit at the median for one feature and diverge on another, so say which."""
    records = [_record(f"{r}8", 2.0) for r in "ABCD"] + [_record("E8", 50.0)]
    out = {r["unit_id"]: r for r in add_cross_record_flags(copy.deepcopy(records))}
    note = out["E8"]["qc"]["note"]
    assert "mfr_hz" in note and "z=+" in note
    assert out["A8"]["qc"].get("note") in (None, "")


def test_small_replicate_groups_are_never_flagged() -> None:
    records = [_record("A8", 2.0), _record("B8", 90.0)]
    out = add_cross_record_flags(copy.deepcopy(records))
    assert all("replicate_divergence" not in r["qc"]["flags"] for r in out)


def test_exposure_without_a_baseline_for_the_same_unit_is_flagged() -> None:
    records = [_record("A8", 2.0, phase="exposure"), _record("B8", 2.0, phase="baseline")]
    out = {r["unit_id"]: r for r in add_cross_record_flags(copy.deepcopy(records))}
    assert "missing_baseline_pair" in out["A8"]["qc"]["flags"]
    assert "missing_baseline_pair" not in out["B8"]["qc"]["flags"]


def test_paired_exposure_is_not_flagged() -> None:
    records = [_record("A8", 2.0, phase="exposure"), _record("A8", 1.0, phase="baseline")]
    out = add_cross_record_flags(copy.deepcopy(records))
    assert all("missing_baseline_pair" not in r["qc"]["flags"] for r in out)


def test_flags_stay_unique_and_records_stay_valid() -> None:
    records = [_record(f"{r}8", 2.0) for r in "ABCD"] + [_record("E8", 50.0)]
    records[-1]["qc"]["flags"] = ["replicate_divergence"]  # already flagged upstream
    out = add_cross_record_flags(copy.deepcopy(records))
    for rec in out:
        assert len(rec["qc"]["flags"]) == len(set(rec["qc"]["flags"]))
        assert validate_record(rec) == [], rec["record_id"]


def test_records_without_features_are_left_alone() -> None:
    curve = _record("A8", 2.0)
    curve["data_level"] = "fitted_curve"
    del curve["features"], curve["feature_method"], curve["recording"]
    curve["curve"] = {"endpoint": "acute_firing_rate_mean", "n_concentrations": 7, "hitcall": 0.5}
    out = add_cross_record_flags([copy.deepcopy(curve)])
    assert out[0]["qc"]["flags"] == []
    assert validate_record(out[0]) == []

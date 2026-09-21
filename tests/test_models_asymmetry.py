"""Tests for the per-device test of compartment coupling asymmetry."""

import json

import numpy as np
import pandas as pd
import pytest

from neuromps.models.asymmetry import by_device_type, compare_device_types, device_asymmetry


def _records(region: str, device: str, asymmetries: list[float]) -> pd.DataFrame:
    rows = []
    for i, asym in enumerate(asymmetries):
        rows.append(
            {
                "source": "screenneuropharm_bic",
                "device.device_id": f"Br18-{region}-{device}",
                "device.compartment_id": "red",
                "device.compartment_role": "chamber",
                "compound.dose_value": float(i + 1),
                "compound.dose_unit": "uM",
                "coupling": json.dumps(
                    [
                        {
                            "to_compartment": "green",
                            "method": "x",
                            "asymmetry": asym,
                            "lag_ms": 5.0,
                            "peak_correlation": 0.9,
                        },
                        {
                            "to_compartment": "gray",
                            "method": "x",
                            "asymmetry": 0.9,
                            "lag_ms": 1.0,
                            "peak_correlation": 0.9,
                        },
                    ]
                ),
            }
        )
    return pd.DataFrame(rows)


def test_device_asymmetry_uses_chamber_pairs_only() -> None:
    """The interface strip sits between the chambers; it is not a chamber-to-chamber link."""
    out = device_asymmetry(_records("CxCx", "1", [0.1, 0.2, 0.3]))
    assert len(out) == 1
    assert out.iloc[0]["median_abs_asymmetry"] == pytest.approx(0.2)
    assert out.iloc[0]["n_conditions"] == 3


def test_device_asymmetry_summarizes_magnitude_not_signed_direction() -> None:
    """Which colour is which population is unknown, so the sign is arbitrary per device."""
    positive = device_asymmetry(_records("CxHp", "a", [0.4, 0.5, 0.6]))
    negative = device_asymmetry(_records("CxHp", "b", [-0.4, -0.5, -0.6]))
    assert positive.iloc[0]["median_abs_asymmetry"] == pytest.approx(
        negative.iloc[0]["median_abs_asymmetry"]
    )


def test_one_row_per_device_regardless_of_condition_count() -> None:
    many = pd.concat(
        [_records("CxCx", "1", [0.1] * 9), _records("HpHp", "2", [0.2] * 9)], ignore_index=True
    )
    out = device_asymmetry(many)
    assert len(out) == 2
    assert set(out["device_type"]) == {"CxCx", "HpHp"}


def test_comparison_reports_effect_size_and_p_value() -> None:
    frames = [_records("CxHp", f"h{i}", [0.3, 0.35, 0.4]) for i in range(6)]
    frames += [_records("CxCx", f"s{i}", [0.02, 0.03, 0.04]) for i in range(5)]
    out = compare_device_types(device_asymmetry(pd.concat(frames, ignore_index=True)))
    assert out["n_heterogeneous"] == 6
    assert out["n_homogeneous"] == 5
    assert out["p_value"] < 0.05
    assert out["rank_biserial"] > 0.8
    assert out["median_heterogeneous"] > out["median_homogeneous"]


def test_comparison_is_not_significant_when_groups_match() -> None:
    rng = np.random.default_rng(0)
    frames = [_records("CxHp", f"h{i}", list(rng.normal(0.2, 0.05, 3))) for i in range(6)]
    frames += [_records("CxCx", f"s{i}", list(rng.normal(0.2, 0.05, 3))) for i in range(5)]
    out = compare_device_types(device_asymmetry(pd.concat(frames, ignore_index=True)))
    assert out["p_value"] > 0.05


def test_comparison_refuses_when_a_group_is_too_small() -> None:
    frames = [_records("CxHp", "h0", [0.3]), _records("CxCx", "s0", [0.02])]
    out = compare_device_types(device_asymmetry(pd.concat(frames, ignore_index=True)))
    assert out["p_value"] is None
    assert "too few" in out["note"]


def test_by_device_type_separates_the_two_homogeneous_types() -> None:
    """Pooling CxCx with HpHp could manufacture a result; report each type."""
    rng = np.random.default_rng(3)
    frames = [_records("CxHp", f"h{i}", list(rng.normal(0.30, 0.05, 2))) for i in range(5)]
    frames += [_records("CxCx", f"c{i}", list(rng.normal(0.02, 0.01, 2))) for i in range(5)]
    frames += [_records("HpHp", f"p{i}", list(rng.normal(0.29, 0.05, 2))) for i in range(5)]
    out = by_device_type(device_asymmetry(pd.concat(frames, ignore_index=True)))
    assert set(out["by_type"]) == {"CxHp", "CxCx", "HpHp"}
    assert out["by_type"]["CxCx"]["median"] < out["by_type"]["HpHp"]["median"]
    assert out["pairwise"]["CxCx_vs_CxHp"] < 0.05
    assert out["pairwise"]["CxHp_vs_HpHp"] > 0.05  # these two do not differ
    assert "kruskal_p" in out


def test_holm_adjusts_for_the_number_of_pairwise_tests() -> None:
    from neuromps.models.asymmetry import holm

    adjusted = holm({"a": 0.01, "b": 0.04, "c": 0.5})
    assert adjusted["a"] == pytest.approx(0.03)  # 0.01 * 3
    assert adjusted["b"] == pytest.approx(0.08)  # 0.04 * 2
    assert adjusted["c"] == pytest.approx(0.5)
    assert adjusted["a"] <= adjusted["b"] <= adjusted["c"]  # monotone by construction


def test_main_skips_an_asset_built_without_the_two_chamber_source(tmp_path) -> None:
    """A --quick run has no compartment records, and no device columns at all."""
    from neuromps.models.asymmetry import main

    asset = tmp_path / "neuromps.parquet"
    pd.DataFrame([{"source": "kapucu2022", "features.mfr_hz": 1.0}]).to_parquet(asset)
    assert main(["--asset", str(asset)]) == 0

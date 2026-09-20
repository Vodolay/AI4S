"""Tests for directed coupling between compartments, on constructed signals."""

import numpy as np
import pytest

from neuromps.features.coupling import CouplingParams, directed_coupling, population_rate


def poisson(rate_hz: float, duration_s: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sort(rng.uniform(0, duration_s, int(rate_hz * duration_s)))


def relay(source: np.ndarray, delay_s: float, fraction: float, seed: int) -> np.ndarray:
    """A downstream copy: `fraction` of source spikes, delayed."""
    rng = np.random.default_rng(seed)
    keep = source[rng.random(source.size) < fraction]
    return np.sort(keep + delay_s)


def test_population_rate_bins_all_electrodes_together() -> None:
    rate, bin_s = population_rate({"a": np.array([0.01, 0.02]), "b": np.array([0.01])}, 1.0, 0.01)
    assert bin_s == 0.01
    assert rate.sum() == 3
    assert rate[1] == 2  # two spikes in the second 10 ms bin


def test_directed_coupling_recovers_the_delay_and_its_sign() -> None:
    src = poisson(30.0, 300.0, seed=0)
    dst = relay(src, delay_s=0.02, fraction=0.6, seed=1)
    out = directed_coupling({"s": src}, {"d": dst}, duration_s=300.0)
    assert out["lag_ms"] == pytest.approx(20.0, abs=5.0)
    assert out["asymmetry"] > 0.2  # source leads target


def test_directed_coupling_is_antisymmetric_when_arguments_swap() -> None:
    src = poisson(30.0, 300.0, seed=2)
    dst = relay(src, delay_s=0.02, fraction=0.6, seed=3)
    forward = directed_coupling({"s": src}, {"d": dst}, duration_s=300.0)
    backward = directed_coupling({"d": dst}, {"s": src}, duration_s=300.0)
    assert backward["lag_ms"] == pytest.approx(-forward["lag_ms"], abs=5.0)
    assert backward["asymmetry"] == pytest.approx(-forward["asymmetry"], abs=0.05)


def test_independent_compartments_report_no_direction_at_all() -> None:
    """Uncoupled compartments must not yield a confident direction from noise."""
    a, b = poisson(30.0, 300.0, seed=4), poisson(30.0, 300.0, seed=5)
    out = directed_coupling({"a": a}, {"b": b}, duration_s=300.0)
    assert out["peak_correlation"] < 0.05
    assert out["asymmetry"] is None
    assert out["lag_ms"] is None


def test_symmetric_coupling_gives_asymmetry_near_zero() -> None:
    """Both directions relayed equally: coupled, but with no net direction."""
    common = poisson(30.0, 300.0, seed=6)
    a = np.sort(np.concatenate([common, poisson(10.0, 300.0, seed=7)]))
    b = np.sort(np.concatenate([common, poisson(10.0, 300.0, seed=8)]))
    out = directed_coupling({"a": a}, {"b": b}, duration_s=300.0)
    assert out["peak_correlation"] > 0.3
    assert abs(out["asymmetry"]) < 0.15
    assert abs(out["lag_ms"]) <= 10.0


def test_empty_compartment_returns_nulls_not_zeros() -> None:
    out = directed_coupling({"a": poisson(20.0, 60.0, 9)}, {"b": np.array([])}, duration_s=60.0)
    assert out["asymmetry"] is None
    assert out["lag_ms"] is None
    assert out["peak_correlation"] is None


def test_params_are_reported_for_provenance() -> None:
    method = CouplingParams().as_method_record()
    assert method["coupling_method"].startswith("binned_cross_correlation")
    assert "bin=" in method["coupling_method"]

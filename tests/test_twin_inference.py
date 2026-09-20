"""Tests for simulation-based inference: prior, summaries, cache and training."""

from pathlib import Path

import numpy as np
import pytest

from neuromps.twin.inference import (
    PARAMETER_NAMES,
    PRIOR_BOUNDS,
    SUMMARY_NAMES,
    load_simulations,
    params_from_vector,
    prior_bounds_tensor,
    save_simulations,
    simulate_summary,
    summary_vector,
)


def test_prior_covers_every_inferred_parameter() -> None:
    assert set(PRIOR_BOUNDS) == set(PARAMETER_NAMES)
    for name, (low, high) in PRIOR_BOUNDS.items():
        assert low < high, name


def test_prior_bounds_tensor_matches_the_declared_order() -> None:
    low, high = prior_bounds_tensor()
    assert list(low.shape) == [len(PARAMETER_NAMES)]
    for i, name in enumerate(PARAMETER_NAMES):
        assert float(low[i]) == pytest.approx(PRIOR_BOUNDS[name][0], abs=1e-6)
        assert float(high[i]) == pytest.approx(PRIOR_BOUNDS[name][1], abs=1e-6)


def test_parameter_vector_round_trips_into_network_parameters() -> None:
    vector = np.array([np.log10(8.0), 12.0, 0.5, 0.8, 1.2])
    params = params_from_vector(vector, n_per_compartment=60, duration_s=1.0, seed=0)
    assert params.forward_ratio == pytest.approx(8.0)
    assert params.inter_in_degree == pytest.approx(12.0)
    assert params.gabaa_scale == pytest.approx(0.5)
    assert params.threshold_scale == pytest.approx(1.2)


def test_forward_ratio_is_sampled_in_log_space() -> None:
    """A ratio of 1 and a ratio of 20 must be equally reachable."""
    low, high = PRIOR_BOUNDS["log10_forward_ratio"]
    assert 10**low == pytest.approx(1.0)
    assert 10**high >= 10.0


def test_summary_vector_has_a_fixed_finite_length() -> None:
    features = {0: {"mfr_hz": 2.0, "synchrony_index": 0.4}, 1: {"mfr_hz": 1.0}}
    coupling = {"peak_correlation": 0.5, "lag_ms": 5.0, "asymmetry": 0.2}
    vector = summary_vector(features, coupling)
    assert vector.shape == (len(SUMMARY_NAMES),)
    assert np.all(np.isfinite(vector))


def test_summary_includes_the_rate_imbalance_term() -> None:
    """HD11: coupling asymmetry tracks relative rate, so rate imbalance must be visible."""
    assert "log_rate_ratio" in SUMMARY_NAMES
    balanced = summary_vector({0: {"mfr_hz": 2.0}, 1: {"mfr_hz": 2.0}}, {})
    tilted = summary_vector({0: {"mfr_hz": 4.0}, 1: {"mfr_hz": 1.0}}, {})
    index = SUMMARY_NAMES.index("log_rate_ratio")
    assert balanced[index] == pytest.approx(0.0)
    assert tilted[index] > 1.0


def test_missing_measurements_become_sentinels_not_nan() -> None:
    vector = summary_vector({0: {}, 1: {}}, {})
    assert np.all(np.isfinite(vector))
    assert "coupling_measured" in SUMMARY_NAMES
    assert vector[SUMMARY_NAMES.index("coupling_measured")] == 0.0


def test_simulate_summary_is_reproducible_and_finite() -> None:
    theta = np.array([np.log10(2.0), 8.0, 1.0, 1.0, 1.0])
    a = simulate_summary(theta, n_per_compartment=60, duration_s=1.0, seed=1)
    b = simulate_summary(theta, n_per_compartment=60, duration_s=1.0, seed=1)
    assert np.allclose(a, b)
    assert np.all(np.isfinite(a))


def test_simulations_round_trip_through_the_cache(tmp_path: Path) -> None:
    theta = np.random.default_rng(0).normal(size=(7, len(PARAMETER_NAMES)))
    x = np.random.default_rng(1).normal(size=(7, len(SUMMARY_NAMES)))
    path = tmp_path / "sims.npz"
    save_simulations(path, theta, x, metadata={"n_per_compartment": 60})
    loaded = load_simulations(path)
    assert np.allclose(loaded["theta"], theta)
    assert np.allclose(loaded["x"], x)
    assert loaded["metadata"]["n_per_compartment"] == 60


def test_cache_records_the_summary_and_parameter_names(tmp_path: Path) -> None:
    """A cache written under a different summary definition must be detectable."""
    path = tmp_path / "sims.npz"
    save_simulations(path, np.zeros((2, 5)), np.zeros((2, len(SUMMARY_NAMES))), metadata={})
    loaded = load_simulations(path)
    assert list(loaded["parameter_names"]) == list(PARAMETER_NAMES)
    assert list(loaded["summary_names"]) == list(SUMMARY_NAMES)


def test_ranks_are_bounded_by_the_sample_count() -> None:
    """Rank statistics must lie in [0, n_samples]; anything else is a coding error."""

    class _StubPosterior:
        def sample(self, shape, x, show_progress_bars=False):  # noqa: ARG002
            import torch

            return torch.zeros((shape[0], len(PARAMETER_NAMES)))

    from neuromps.twin.inference import posterior_ranks

    theta = np.ones((3, len(PARAMETER_NAMES)))
    x = np.zeros((3, len(SUMMARY_NAMES)))
    ranks = posterior_ranks(_StubPosterior(), theta, x, n_samples=50)
    assert ranks.shape == (3, len(PARAMETER_NAMES))
    assert ranks.min() >= 0 and ranks.max() <= 50
    assert (ranks == 50).all()  # every sample below the true value


def test_contraction_is_zero_when_the_posterior_equals_the_prior() -> None:
    """A posterior as wide as the prior has learned nothing; that must read as 0."""
    import torch

    from neuromps.twin.inference import PRIOR_BOUNDS, contraction

    class _PriorWidePosterior:
        def sample(self, shape, x, show_progress_bars=False):  # noqa: ARG002
            rng = np.random.default_rng(0)
            columns = [
                rng.uniform(PRIOR_BOUNDS[name][0], PRIOR_BOUNDS[name][1], shape[0])
                for name in PARAMETER_NAMES
            ]
            return torch.as_tensor(np.stack(columns, axis=1), dtype=torch.float32)

    out = contraction(
        _PriorWidePosterior(), np.zeros((2, 5)), np.zeros((2, len(SUMMARY_NAMES))), 500
    )
    for name in PARAMETER_NAMES:
        assert abs(out[name]["contraction"]) < 0.15, name


def test_draws_outside_a_tightened_prior_are_rejected() -> None:
    """A cache made under a wider prior stays usable by rejection, not by pretending."""
    from neuromps.twin.inference import within_prior

    inside = np.array([[0.5, 10.0, 1.0, 1.0, 1.0]])
    outside = np.array([[0.5, 10.0, 1.0, 1.0, 1.49]])  # threshold beyond the current bound
    assert within_prior(inside).all()
    assert not within_prior(outside).any()


def test_threshold_prior_stops_below_the_silence_cliff() -> None:
    """Above ~1.12 the twin is silent in 98% of draws and carries no information."""
    assert PRIOR_BOUNDS["threshold_scale"][1] <= 1.2


def test_learning_curve_skips_sizes_larger_than_the_pool() -> None:
    from neuromps.twin.inference import learning_curve

    theta = np.random.default_rng(0).uniform(size=(30, len(PARAMETER_NAMES)))
    x = np.random.default_rng(1).uniform(size=(30, len(SUMMARY_NAMES)))
    rows = learning_curve(theta, x, sizes=(5000,), holdout=10)
    assert rows == []  # nothing trainable, and no crash


def test_rank_uniformity_reports_a_statistic_not_only_a_p_value() -> None:
    """A p-value alone is not comparable across evaluation-set sizes."""
    from neuromps.twin.inference import _rank_uniformity

    rng = np.random.default_rng(0)
    ranks = rng.integers(0, 100, size=(200, len(PARAMETER_NAMES)))
    out = _rank_uniformity(ranks, 100)
    for name in PARAMETER_NAMES:
        assert {"ks_statistic", "ks_p", "n_evaluated"} <= set(out[name])
        assert 0.0 <= out[name]["ks_statistic"] <= 1.0
        assert out[name]["n_evaluated"] == 200


def test_default_layout_matches_the_real_chamber_electrode_count() -> None:
    """Simulated and real geometry must agree, or electrode-count summaries go OOD."""
    from neuromps.twin.inference import default_layout

    assert default_layout().electrodes_per_compartment == 48


def test_a_cache_from_a_different_layout_is_refused() -> None:
    """Summary names alone do not pin down what the numbers mean; geometry does too."""
    from neuromps.twin.inference import check_cache_matches_code, summary_pipeline_fingerprint

    good = {
        "summary_names": list(SUMMARY_NAMES),
        "metadata": {
            "electrodes_per_compartment": 48,
            "summary_pipeline_sha256": summary_pipeline_fingerprint(),
        },
    }
    check_cache_matches_code(good)  # does not raise
    for bad in ({"electrodes_per_compartment": 24}, {}):
        with pytest.raises(SystemExit, match="electrodes per compartment"):
            check_cache_matches_code({"summary_names": list(SUMMARY_NAMES), "metadata": bad})


def test_saving_records_the_layout_that_produced_the_summaries(tmp_path: Path) -> None:
    path = tmp_path / "sims.npz"
    save_simulations(path, np.zeros((2, 5)), np.zeros((2, len(SUMMARY_NAMES))), metadata={})
    assert load_simulations(path)["metadata"]["electrodes_per_compartment"] == 48


def test_a_cache_from_different_summary_code_is_refused() -> None:
    """HD16 changed what coupling means without any guard noticing; this is that guard."""
    from neuromps.twin.inference import check_cache_matches_code, summary_pipeline_fingerprint

    assert summary_pipeline_fingerprint() == summary_pipeline_fingerprint()  # stable
    stale = {
        "summary_names": list(SUMMARY_NAMES),
        "metadata": {
            "electrodes_per_compartment": 48,
            "summary_pipeline_sha256": "0000000000000000",
        },
    }
    with pytest.raises(SystemExit, match="summary code"):
        check_cache_matches_code(stale)

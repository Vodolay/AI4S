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
    from neuromps.twin.inference import (
        calibration_fingerprint,
        check_cache_matches_code,
        summary_pipeline_fingerprint,
    )

    good = {
        "summary_names": list(SUMMARY_NAMES),
        "metadata": {
            "electrodes_per_compartment": 48,
            "summary_pipeline_sha256": summary_pipeline_fingerprint(),
            "calibration_sha256": calibration_fingerprint(),
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


def test_training_is_seeded_before_the_posterior_is_built() -> None:
    """An unseeded run moved a reported p-value from 0.46 to 0.01 between runs."""
    import inspect

    from neuromps.twin.inference import main

    source = inspect.getsource(main)
    seed_line = source.index("torch.manual_seed(args.seed)")
    train_line = source.index("train_posterior(")
    assert seed_line < train_line, "the seed must be set before training"


def test_an_explicit_network_size_still_overrides_the_calibration() -> None:
    """Tests and probes need a small network; the pipeline should not pass one.

    The default is the calibrated density, so nothing in the pipeline silently
    runs a sparser twin than the one 7.3 reports; an explicit size remains
    available for callers that deliberately want a cheap network.
    """
    vector = np.array([np.log10(8.0), 12.0, 0.5, 0.8, 1.2])
    params = params_from_vector(vector, n_per_compartment=60, duration_s=1.0, seed=0)
    assert params.n_per_compartment == 60


def test_inference_uses_the_calibrated_density() -> None:
    """Density is what sets the twin's synchrony regime (7.3).

    Inference used to run a smaller network for tractability, which meant the
    posterior was learned on a network several times sparser than the one the
    calibration fitted. Measuring the cost showed it was 1.55x, not the 6x
    feared, so the gap is closed rather than documented.
    """
    from neuromps.twin.calibrate import calibrated_params
    from neuromps.twin.inference import inference_calibration

    calibrated = calibrated_params().get("n_per_compartment")
    if calibrated is None:
        pytest.skip("no calibration on disk")
    assert inference_calibration().get("n_per_compartment") == calibrated
    vector = np.array([np.log10(8.0), 12.0, 0.5, 0.8, 1.2])
    assert params_from_vector(vector, duration_s=1.0, seed=0).n_per_compartment == calibrated


def test_a_cache_from_a_different_calibration_is_refused(tmp_path: Path) -> None:
    """The failure this prevents is silent: a posterior trained on a dead model.

    Nothing else in the cache changes when the calibration does - same prior,
    same summaries, same layout - so without this the stale simulations are
    reused and the run looks clean.
    """
    from neuromps.twin.inference import calibration_fingerprint, check_cache_matches_code

    path = tmp_path / "sims.npz"
    save_simulations(path, np.zeros((2, 5)), np.zeros((2, len(SUMMARY_NAMES))), metadata={})
    cached = load_simulations(path)
    assert cached["metadata"]["calibration_sha256"] == calibration_fingerprint()
    check_cache_matches_code(cached)  # the matching case must pass

    cached["metadata"]["calibration_sha256"] = "0" * 64
    with pytest.raises(SystemExit, match="calibration"):
        check_cache_matches_code(cached)


def test_a_stale_cache_is_rebuilt_rather_than_killing_the_pipeline(tmp_path, monkeypatch) -> None:
    """`run_all.py` passes no --rebuild, so raising here left the whole run dead.

    The guard still raises for `real_inference`, which *reads* the cache and must
    refuse a superseded one. The writer's answer to a stale artifact is to
    regenerate it.
    """
    from neuromps.twin import inference

    path = tmp_path / "sims.npz"
    save_simulations(path, np.zeros((2, 5)), np.zeros((2, len(SUMMARY_NAMES))), metadata={})
    monkeypatch.setattr(inference, "simulation_cache", lambda: path)
    assert inference.cache_is_usable() is True

    # A cache written under different summary code must be rebuilt, not fatal.
    monkeypatch.setattr(inference, "summary_pipeline_fingerprint", lambda: "different")
    assert inference.cache_is_usable() is False


def test_the_reader_still_refuses_a_stale_cache_outright() -> None:
    """Rebuilding is the writer's answer; a reader must not silently compare."""
    import inspect

    from neuromps.twin import real_inference

    assert "check_cache_matches_code(cached)" in inspect.getsource(real_inference.main)


class _CountingPosterior:
    """Stands in for a trained posterior: records what it was asked for."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def sample(self, shape, x, show_progress_bars=False):  # noqa: ARG002
        import torch

        self.calls.append(shape[0])
        return torch.zeros((shape[0], len(PARAMETER_NAMES)))


def test_amortized_timing_asks_every_observation_for_the_stated_draws() -> None:
    """The report quotes seconds per posterior; it must be per observation, at that size."""
    from neuromps.twin.inference import amortized_timing

    posterior = _CountingPosterior()
    out = amortized_timing(posterior, np.zeros((7, len(SUMMARY_NAMES))), n_draws=250)
    assert posterior.calls == [250] * 7
    assert out["n_observations"] == 7 and out["posterior_draws"] == 250
    assert out["median_seconds_per_posterior"] >= 0
    assert "forward_simulation_seconds" not in out  # nothing simulated, nothing claimed


def test_amortized_timing_times_the_simulator_only_when_given_one() -> None:
    """The comparison is the point: a posterior against one run of the simulator."""
    from neuromps.twin.inference import amortized_timing

    ran = []
    out = amortized_timing(
        _CountingPosterior(),
        np.zeros((2, len(SUMMARY_NAMES))),
        simulate=lambda: ran.append(True),
        simulated_s=10.0,
    )
    assert ran == [True]
    assert out["forward_simulation_seconds"] >= 0
    assert out["forward_simulated_duration_s"] == 10.0


def test_a_full_run_records_the_timing_with_the_other_results() -> None:
    """Recorded only by a separate command, a full rerun would silently drop it."""
    import inspect

    from neuromps.twin import inference

    assert '"amortized_inference": amortized_timing(' in inspect.getsource(inference.main)


def test_the_quick_tier_times_a_token_sample_and_no_simulation() -> None:
    """Its five-epoch posterior rejection-samples slowly: twenty 1,000-draw posteriors
    from it ran a fresh clone's quick run far past its ten minutes, and the quick
    tier's timings are never reported."""
    from neuromps.twin.inference import TIMING_DRAWS, _timing_arguments

    quick = _timing_arguments(quick=True)
    assert "simulate" not in quick
    assert quick["n_observations"] <= 2 and quick["n_draws"] < TIMING_DRAWS


def test_training_starts_from_the_same_random_state_with_or_without_the_cache(
    tmp_path, monkeypatch
) -> None:
    """A fresh clone's full run matched the report; rerunning it from the cache did not.

    Drawing the parameters consumes torch's random stream, and only the fresh path
    drew them, so a cached rerun trained a different network and moved every
    calibration statistic. Both paths must hand training the same state.
    """
    import torch

    from neuromps.twin import inference

    monkeypatch.setattr(inference, "simulation_cache", lambda: tmp_path / "sims.npz")
    monkeypatch.setattr(inference, "twin_dir", lambda: tmp_path)
    monkeypatch.setattr(inference, "cache_is_usable", lambda: True)

    def fake_simulations(n, duration_s=10.0, seed=0, **kwargs):
        theta = inference.draw_parameters(n, seed)
        return theta, np.zeros((n, len(SUMMARY_NAMES)))

    states = []

    class _Stop(Exception):
        pass

    def capture(theta, x, max_epochs=None):
        states.append(torch.get_rng_state().clone())
        raise _Stop

    monkeypatch.setattr(inference, "run_simulations", fake_simulations)
    monkeypatch.setattr(inference, "train_posterior", capture)
    for _ in range(2):  # the first run simulates and writes the cache; the second reuses it
        with pytest.raises(_Stop):
            inference.main(["--simulations", "40"])
    assert len(states) == 2
    assert torch.equal(states[0], states[1])

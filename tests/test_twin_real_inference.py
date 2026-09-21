"""Tests for applying the trained posterior to real recordings."""

import json

import numpy as np
import pandas as pd
import pytest

from neuromps.twin.inference import SUMMARY_NAMES
from neuromps.twin.real_inference import (
    distribution_check,
    dose_trend,
    real_summaries,
)


def _records() -> pd.DataFrame:
    rows = []
    for device, dose, unit in (("d1", None, None), ("d1", 10.0, "uM"), ("d2", 30.0, "uM")):
        for compartment, mfr in (("red", 4.0), ("green", 2.0)):
            coupling = [
                {
                    "to_compartment": "green" if compartment == "red" else "red",
                    "method": "x",
                    "peak_correlation": 0.8,
                    "lag_ms": 5.0,
                    "asymmetry": 0.1,
                },
                {
                    "to_compartment": "gray",
                    "method": "x",
                    "peak_correlation": 0.5,
                    "lag_ms": 0.0,
                    "asymmetry": 0.0,
                },
            ]
            rows.append(
                {
                    "source": "screenneuropharm_bic",
                    "data_level": "spike_times",
                    "device.device_id": device,
                    "device.compartment_id": compartment,
                    "compound.dose_value": dose,
                    "compound.dose_unit": unit,
                    "recording.phase": "baseline" if dose is None else "exposure",
                    "qc.usable": True,
                    "coupling": json.dumps(coupling),
                    "features.mfr_hz": mfr,
                    "features.burst_rate_per_min": 10.0,
                    "features.pct_spikes_in_bursts": 80.0,
                    "features.network_burst_rate_per_min": 15.0,
                    "features.synchrony_index": 0.7,
                    "features.isi_cv": 1.5,
                    "features.active_electrodes": 40,
                }
            )
    return pd.DataFrame(rows)


def test_one_summary_row_per_device_condition() -> None:
    out = real_summaries(_records())
    assert len(out) == 3
    assert set(out["device_id"]) == {"d1", "d2"}


def test_summary_columns_match_the_simulator_exactly() -> None:
    """A real vector fed to the posterior must be built like the simulated ones."""
    out = real_summaries(_records())
    assert [c for c in out.columns if c in SUMMARY_NAMES] == list(SUMMARY_NAMES)


def test_chambers_map_to_compartments_and_rate_ratio_follows() -> None:
    out = real_summaries(_records()).iloc[0]
    assert out["c0_mfr_hz"] == 4.0  # red
    assert out["c1_mfr_hz"] == 2.0  # green
    assert out["log_rate_ratio"] == pytest.approx(1.0, abs=0.01)


def test_recordings_without_both_chambers_are_dropped() -> None:
    records = _records()
    records = records[
        ~((records["device.device_id"] == "d2") & (records["device.compartment_id"] == "green"))
    ]
    assert len(real_summaries(records)) == 2


def test_distribution_check_flags_dimensions_outside_the_simulated_range() -> None:
    rng = np.random.default_rng(0)
    simulated = rng.normal(0, 1, size=(500, len(SUMMARY_NAMES)))
    real = np.zeros((10, len(SUMMARY_NAMES)))
    real[:, 3] = 50.0  # far outside
    out = distribution_check(real, simulated)
    assert out["by_dimension"][SUMMARY_NAMES[3]]["fraction_inside"] == 0.0
    assert out["by_dimension"][SUMMARY_NAMES[0]]["fraction_inside"] == 1.0
    assert SUMMARY_NAMES[3] in out["outside_dimensions"]


def test_distribution_check_reports_an_overall_verdict() -> None:
    rng = np.random.default_rng(1)
    simulated = rng.normal(0, 1, size=(300, len(SUMMARY_NAMES)))
    inside = rng.normal(0, 1, size=(20, len(SUMMARY_NAMES)))
    out = distribution_check(inside, simulated)
    assert 0.0 <= out["mean_fraction_inside"] <= 1.0
    assert out["mean_fraction_inside"] > 0.8


def test_dose_trend_reports_direction_and_significance() -> None:
    doses = np.array([0.0, 0.1, 1.0, 10.0, 100.0, 300.0])
    falling = np.array([1.0, 0.9, 0.8, 0.6, 0.4, 0.3])
    out = dose_trend(doses, falling)
    assert out["spearman_rho"] < -0.9
    assert out["naive_p_value"] < 0.05
    assert out["direction"] == "falls with dose"


def test_dose_trend_without_groups_says_the_p_value_is_naive() -> None:
    """One device contributes a whole series, so ungrouped significance is overstated."""
    out = dose_trend(np.array([0.0, 1.0, 10.0, 100.0]), np.array([1.0, 0.8, 0.6, 0.4]))
    assert "naive" in out["note"]
    assert "spearman_rho_ci95" not in out


def test_dose_trend_with_groups_reports_a_cluster_interval() -> None:
    doses = np.tile([0.0, 1.0, 10.0, 100.0], 4)
    values = np.tile([1.0, 0.8, 0.6, 0.4], 4)
    groups = np.repeat(["d1", "d2", "d3", "d4"], 4)
    out = dose_trend(doses, values, groups=groups, n_resamples=200)
    low, high = out["spearman_rho_ci95"]
    assert low <= out["spearman_rho"] <= high
    assert out["n_devices"] == 4
    assert out["excludes_zero"] is True


def test_sensitivity_replaces_only_the_named_dimensions() -> None:
    from neuromps.twin.real_inference import sensitivity_without_dimensions

    captured = {}

    class _StubPosterior:
        def sample(self, shape, x, show_progress_bars=False, **kwargs):  # noqa: ARG002
            import torch

            captured["x"] = np.asarray(x)
            return torch.zeros((shape[0], 5))

    real = np.ones((1, len(SUMMARY_NAMES)))
    simulated = np.full((10, len(SUMMARY_NAMES)), 7.0)
    sensitivity_without_dimensions(_StubPosterior(), real, simulated, ["c0_mfr_hz"], n_samples=3)
    index = SUMMARY_NAMES.index("c0_mfr_hz")
    assert captured["x"][index] == 7.0
    assert captured["x"][index + 1] == 1.0  # untouched


def test_dose_trend_needs_enough_distinct_doses() -> None:
    out = dose_trend(np.array([1.0, 1.0]), np.array([0.5, 0.4]))
    assert out["spearman_rho"] is None
    assert "too few" in out["note"]


def test_a_bimodal_dimension_passes_the_range_test_but_is_flagged_unsupported() -> None:
    """Silent-or-saturated simulations bracket the real values without covering them."""
    simulated = np.zeros((400, len(SUMMARY_NAMES)))
    index = SUMMARY_NAMES.index("c0_active_electrodes")
    simulated[:, index] = np.repeat([0.0, 48.0], 200)  # two modes, nothing between
    real = np.zeros((20, len(SUMMARY_NAMES)))
    real[:, index] = 43.0  # inside [0, 48], but where no simulation lands
    out = distribution_check(real, simulated)
    assert out["by_dimension"]["c0_active_electrodes"]["fraction_inside"] == 1.0
    assert out["by_dimension"]["c0_active_electrodes"]["simulated_support_in_real_iqr"] == 0.0
    assert "c0_active_electrodes" not in out["outside_dimensions"]
    assert "c0_active_electrodes" in out["unsupported_dimensions"]


class _RecordingPosterior:
    """Stub that returns draws half inside and half outside the prior box."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def sample(self, shape, x, show_progress_bars=False, **kwargs):  # noqa: ARG002
        import torch

        from neuromps.twin.inference import PRIOR_BOUNDS

        self.calls.append(kwargs)
        n = shape[0]
        inside = np.array([[np.mean(PRIOR_BOUNDS[k]) for k in PRIOR_BOUNDS]] * (n // 2))
        outside = np.array([[PRIOR_BOUNDS[k][1] + 10.0 for k in PRIOR_BOUNDS]] * (n - n // 2))
        return torch.as_tensor(np.vstack([inside, outside]), dtype=torch.float32)


def test_prior_acceptance_measures_draws_landing_inside_the_prior() -> None:
    from neuromps.twin.real_inference import prior_acceptance

    posterior = _RecordingPosterior()
    out = prior_acceptance(posterior, np.zeros((3, len(SUMMARY_NAMES))), n_draws=100)
    assert out["median_acceptance"] == pytest.approx(0.5)
    assert out["n_draws_per_recording"] == 100
    assert all(call.get("reject_outside_prior") is False for call in posterior.calls)


def test_inference_is_time_capped_and_records_how_many_samples_it_got() -> None:
    """An unbounded rejection sampler can hang for minutes on a misspecified input."""
    from neuromps.twin.real_inference import infer_parameters

    posterior = _RecordingPosterior()
    rows = infer_parameters(posterior, np.zeros((2, len(SUMMARY_NAMES))), n_samples=50)
    assert rows[0]["n_effective_samples"] == 50
    assert all(call.get("max_sampling_time") is not None for call in posterior.calls)
    assert all(call.get("return_partial_on_timeout") is True for call in posterior.calls)


def test_the_figure_caption_quotes_the_cluster_interval_not_the_naive_p() -> None:
    """113 recordings from 16 devices: the naive p must not be what a reader sees."""
    from neuromps.twin.real_inference import trend_caption

    doses = np.tile([0.0, 1.0, 10.0, 100.0], 4)
    values = np.tile([1.0, 0.8, 0.6, 0.4], 4)
    groups = np.repeat(["d1", "d2", "d3", "d4"], 4)
    caption = trend_caption(dose_trend(doses, values, groups=groups, n_resamples=200))
    assert "95% CI" in caption
    assert "devices" in caption
    assert "p =" not in caption
    assert "None" not in trend_caption(dose_trend(np.array([1.0, 1.0]), np.array([0.5, 0.4])))


def test_baseline_offset_flags_an_untreated_value_away_from_neutral() -> None:
    """Untreated chips must infer a knob near 1, or the knob is absorbing mismatch."""
    from neuromps.twin.real_inference import baseline_offset

    doses = np.tile([0.0, 0.0, 10.0, 100.0], 4)
    values = np.tile([0.4, 0.45, 0.4, 0.4], 4)
    groups = np.repeat(["d1", "d2", "d3", "d4"], 4)
    out = baseline_offset(values, doses, groups=groups, n_resamples=200)
    assert out["n_untreated"] == 8
    assert out["median"] == pytest.approx(0.425, abs=0.05)
    assert out["includes_neutral"] is False  # 1.0 well outside the interval


def test_baseline_offset_accepts_a_posterior_that_returns_to_neutral() -> None:
    from neuromps.twin.real_inference import baseline_offset

    rng = np.random.default_rng(0)
    doses = np.tile([0.0, 0.0, 10.0], 8)
    values = np.concatenate(
        [[1.0 + rng.normal(0, 0.1), 1.0 + rng.normal(0, 0.1), 0.5] for _ in range(8)]
    )
    groups = np.repeat([f"d{i}" for i in range(8)], 3)
    out = baseline_offset(values, doses, groups=groups, n_resamples=200)
    assert out["includes_neutral"] is True


def test_posterior_sampling_is_seeded_so_reruns_agree() -> None:
    """Unseeded sampling moved the reported rho between runs of the same code."""
    import torch

    from neuromps.twin.real_inference import infer_parameters

    class _NoisyPosterior:
        def sample(self, shape, x, show_progress_bars=False, **kwargs):  # noqa: ARG002
            return torch.rand((shape[0], 5))

    torch.manual_seed(7)
    first = infer_parameters(_NoisyPosterior(), np.zeros((2, len(SUMMARY_NAMES))), n_samples=64)
    torch.manual_seed(7)
    second = infer_parameters(_NoisyPosterior(), np.zeros((2, len(SUMMARY_NAMES))), n_samples=64)
    assert first == second


def test_no_summaries_when_the_asset_has_no_compartment_records() -> None:
    """The two-chamber source is --full only; --quick must not raise on its absence."""
    quick_asset = _records().drop(columns=["device.compartment_id", "device.device_id"])
    assert real_summaries(quick_asset).empty


def test_the_empty_summary_table_still_has_its_columns() -> None:
    """Consumers index the result by column name; the empty case must survive that."""
    quick_asset = _records().drop(columns=["device.compartment_id", "device.device_id"])
    summaries = real_summaries(quick_asset)
    assert summaries.empty
    assert set(SUMMARY_NAMES) <= set(summaries.columns)
    assert summaries[list(SUMMARY_NAMES)].to_numpy(dtype=float).shape == (0, len(SUMMARY_NAMES))

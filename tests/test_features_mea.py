"""Tests for the shared MEA feature extractor, on spike trains with known statistics.

`features/mea.py` is the single extractor: the ETL and the digital twin both call
it, so these tests pin the definitions rather than the implementation details.
"""

import json

import numpy as np
import pytest

from neuromps.features.mea import FeatureParams, extract_features
from neuromps.qc.schema import SCHEMA_PATH


def regular_train(rate_hz: float, duration_s: float, offset: float = 0.0) -> np.ndarray:
    """Perfectly regular spike train (ISI CV = 0)."""
    return np.arange(offset, duration_s, 1.0 / rate_hz)


def bursty_train(n_bursts: int, spikes_per_burst: int, intra_isi: float, period: float):
    """`n_bursts` tight bursts, one per `period` seconds."""
    return np.concatenate(
        [b * period + np.arange(spikes_per_burst) * intra_isi for b in range(n_bursts)]
    )


def test_mean_firing_rate_of_regular_train() -> None:
    f = extract_features({"e1": regular_train(2.0, 60.0)}, duration_s=60.0)
    assert f["mfr_hz"] == pytest.approx(2.0, abs=0.05)
    assert f["active_electrodes"] == 1


def test_isi_cv_is_zero_for_regular_and_near_one_for_poisson() -> None:
    regular = extract_features({"e1": regular_train(5.0, 120.0)}, duration_s=120.0)
    assert regular["isi_cv"] == pytest.approx(0.0, abs=1e-9)
    rng = np.random.default_rng(0)
    poisson = np.cumsum(rng.exponential(1 / 5.0, size=6000))
    poisson = poisson[poisson < 1200.0]
    f = extract_features({"e1": poisson}, duration_s=1200.0)
    assert f["isi_cv"] == pytest.approx(1.0, abs=0.1)
    assert f["mfr_hz"] == pytest.approx(5.0, rel=0.1)


def test_burst_statistics_are_exact_for_constructed_bursts() -> None:
    train = bursty_train(n_bursts=10, spikes_per_burst=8, intra_isi=0.005, period=6.0)
    f = extract_features({"e1": train}, duration_s=60.0)
    assert f["burst_rate_per_min"] == pytest.approx(10.0)
    assert f["spikes_per_burst"] == pytest.approx(8.0)
    assert f["burst_duration_s"] == pytest.approx(7 * 0.005)
    assert f["pct_spikes_in_bursts"] == pytest.approx(100.0)


def test_isolated_spikes_are_not_bursts() -> None:
    f = extract_features({"e1": regular_train(1.0, 60.0)}, duration_s=60.0)
    assert f["burst_rate_per_min"] == 0.0
    assert f["pct_spikes_in_bursts"] == 0.0
    assert f["burst_duration_s"] is None
    assert f["spikes_per_burst"] is None


def test_burst_needs_minimum_spike_count() -> None:
    """A 3-spike cluster is not a burst at burst_min_spikes=5, though the electrode is active."""
    params = FeatureParams(burst_min_spikes=5)
    train = np.sort(np.concatenate([regular_train(1.0, 60.0), [30.002, 30.004]]))
    f = extract_features({"e1": train}, 60.0, params)
    assert f["active_electrodes"] == 1
    assert f["burst_rate_per_min"] == 0.0
    assert extract_features({"e1": train}, 60.0, FeatureParams(burst_min_spikes=3))[
        "burst_rate_per_min"
    ] == pytest.approx(1.0)


def test_inactive_electrodes_are_excluded_from_active_count_and_rate() -> None:
    spikes = {
        "active": regular_train(2.0, 60.0),
        "silent": np.array([1.0, 30.0]),  # 2 spikes/min < 5 spikes/min threshold
    }
    f = extract_features(spikes, duration_s=60.0)
    assert f["active_electrodes"] == 1
    assert f["mfr_hz"] == pytest.approx(2.0, abs=0.05)


def test_no_active_electrodes_gives_nulls_not_zeros() -> None:
    f = extract_features({"e1": np.array([1.0]), "e2": np.array([])}, duration_s=60.0)
    assert f["active_electrodes"] == 0
    assert f["mfr_hz"] is None
    assert f["synchrony_index"] is None
    assert f["isi_cv"] is None


def test_sttc_is_one_for_identical_trains_and_near_zero_for_independent() -> None:
    train = regular_train(5.0, 120.0)
    same = extract_features({"a": train, "b": train.copy()}, duration_s=120.0)
    assert same["synchrony_index"] == pytest.approx(1.0, abs=1e-6)
    rng = np.random.default_rng(1)
    a = np.sort(rng.uniform(0, 300, 1500))
    b = np.sort(rng.uniform(0, 300, 1500))
    indep = extract_features({"a": a, "b": b}, duration_s=300.0)
    assert abs(indep["synchrony_index"]) < 0.1


def test_network_burst_requires_multiple_electrodes() -> None:
    burst = bursty_train(n_bursts=5, spikes_per_burst=10, intra_isi=0.004, period=12.0)
    together = {f"e{i}": burst + i * 0.001 for i in range(8)}
    f = extract_features(together, duration_s=60.0)
    assert f["network_burst_rate_per_min"] == pytest.approx(5.0)
    rng = np.random.default_rng(7)
    alone = {"e0": burst}
    alone.update({f"e{i}": np.sort(rng.uniform(0, 60, 180)) for i in range(1, 8)})
    assert extract_features(alone, duration_s=60.0)["network_burst_rate_per_min"] == 0.0


def test_synchronous_tonic_firing_counts_as_network_events() -> None:
    """Documents the definition: this is Axion-style coincidence, not rate elevation.

    Electrodes firing in lockstep at 2 Hz produce 120 network events per minute even
    though no electrode is bursting. Consumers comparing against a rate-elevation
    detector must not treat the two as the same quantity (STANDARD.md 4.2).
    """
    lockstep = {f"e{i}": regular_train(2.0, 60.0) for i in range(8)}
    f = extract_features(lockstep, duration_s=60.0)
    assert f["network_burst_rate_per_min"] == pytest.approx(120.0)
    assert f["burst_rate_per_min"] == 0.0


def test_extract_features_is_deterministic_and_does_not_mutate_input() -> None:
    spikes = {"e1": bursty_train(4, 6, 0.006, 10.0)}
    before = {k: v.copy() for k, v in spikes.items()}
    first = extract_features(spikes, 60.0)
    second = extract_features(spikes, 60.0)
    assert first == second
    for k, v in spikes.items():
        assert np.array_equal(v, before[k])


def test_unsorted_spike_times_are_handled() -> None:
    train = bursty_train(3, 6, 0.006, 10.0)
    shuffled = np.random.default_rng(2).permutation(train)
    assert extract_features({"e": shuffled}, 60.0) == extract_features({"e": train}, 60.0)


def test_feature_keys_match_the_schema_exactly() -> None:
    """The extractor and schema/neuromps_schema.json must not drift apart."""
    schema = json.loads(SCHEMA_PATH.read_text())
    expected = set(schema["$defs"]["features"]["properties"])
    assert set(extract_features({"e1": regular_train(2.0, 60.0)}, 60.0)) == expected


def test_params_are_reported_for_provenance() -> None:
    params = FeatureParams()
    d = params.as_method_record()
    assert d["burst_method"].startswith("maxinterval")
    assert d["synchrony_method"] == "sttc"
    assert d["active_electrode_threshold_spikes_per_min"] == params.active_min_spikes_per_min


def test_duration_must_be_positive() -> None:
    with pytest.raises(ValueError, match="duration"):
        extract_features({"e1": regular_train(2.0, 10.0)}, duration_s=0.0)


def test_independent_poisson_electrodes_give_no_network_bursts() -> None:
    """Chance-coincidence control: 16 independent 5 Hz electrodes are not a network."""
    rng = np.random.default_rng(3)
    spikes = {f"e{i}": np.sort(rng.uniform(0, 60, 300)) for i in range(16)}
    f = extract_features(spikes, duration_s=60.0)
    assert f["active_electrodes"] == 16
    assert f["network_burst_rate_per_min"] == 0.0


def test_network_bursts_survive_on_top_of_high_background_rate() -> None:
    """The control must not suppress real synchrony: 10 Hz noise + 6 synchronous events."""
    rng = np.random.default_rng(4)
    events = np.arange(5.0, 60.0, 9.0)
    spikes = {}
    for i in range(16):
        noise = np.sort(rng.uniform(0, 60, 600))
        synchronous = np.concatenate([e + np.arange(8) * 0.003 for e in events])
        spikes[f"e{i}"] = np.sort(np.concatenate([noise, synchronous + i * 0.0002]))
    f = extract_features(spikes, duration_s=60.0)
    assert f["network_burst_rate_per_min"] == pytest.approx(len(events), abs=1.0)


# ------------------------------------------------- definition options (D33, post hoc ladder)


def _mixed_plate() -> dict[str, np.ndarray]:
    """Two bursting electrodes, one tonic non-bursting one, one at exactly 5 spikes/min."""
    return {
        "a": bursty_train(n_bursts=10, spikes_per_burst=8, intra_isi=0.005, period=6.0),
        "b": 0.3 + bursty_train(n_bursts=5, spikes_per_burst=6, intra_isi=0.01, period=12.0),
        "c": np.sort((np.arange(1, 41) ** 1.5) % 60.0),
        "d": np.arange(0.0, 60.0, 12.0),
    }


# Recorded from the extractor before the options existed: the defaults must keep
# every value bit for bit, because the asset and the twin were built with them.
GOLDEN_DEFAULTS = {
    "mfr_hz": 0.6458333333333334,
    "burst_rate_per_min": 3.75,
    "burst_duration_s": 0.03999999999999885,
    "spikes_per_burst": 7.333333333333333,
    "pct_spikes_in_bursts": 70.96774193548387,
    "network_burst_rate_per_min": 1.0,
    "synchrony_index": 0.12985001736894017,
    "isi_cv": 1.5459609695883432,
    "active_electrodes": 4,
}
GOLDEN_METHOD_RECORD = {
    "computed_by": "neuromps.features.mea",
    "burst_method": "maxinterval_simple(isi<=0.1s,>=5spikes)",
    "network_burst_method": (
        "binned_participation(bin=0.025s,>=max(3,0.25*active,chance)electrodes,"
        "chance_fp<0.05/recording)"
    ),
    "synchrony_method": "sttc",
    "active_electrode_threshold_spikes_per_min": 5.0,
}


def test_default_outputs_are_unchanged_by_the_definition_options() -> None:
    assert extract_features(_mixed_plate(), 60.0) == GOLDEN_DEFAULTS
    explicit = FeatureParams(active_strict=False, burst_rate_denominator="active")
    assert extract_features(_mixed_plate(), 60.0, explicit) == GOLDEN_DEFAULTS


def test_method_record_is_unchanged_at_defaults() -> None:
    assert FeatureParams().as_method_record() == GOLDEN_METHOD_RECORD


def test_method_record_names_a_non_default_definition() -> None:
    record = FeatureParams(active_strict=True, burst_rate_denominator="bursting").as_method_record()
    assert record["active_electrode_threshold_strict"] is True
    assert record["burst_rate_denominator"] == "bursting"
    only_one = FeatureParams(active_strict=True).as_method_record()
    assert "burst_rate_denominator" not in only_one


def test_strict_threshold_drops_an_electrode_exactly_at_the_threshold() -> None:
    """Electrode d fires exactly 5 spikes in a minute: active under >=, not under >."""
    inclusive = extract_features(_mixed_plate(), 60.0)
    strict = extract_features(_mixed_plate(), 60.0, FeatureParams(active_strict=True))
    assert inclusive["active_electrodes"] == 4
    assert strict["active_electrodes"] == 3
    assert strict["mfr_hz"] == pytest.approx((80 + 30 + 40) / 3 / 60.0)
    assert strict["burst_rate_per_min"] == pytest.approx(15 / 3)


def test_strict_threshold_still_requires_a_spike_at_threshold_zero() -> None:
    p = FeatureParams(active_min_spikes_per_min=0.0, active_strict=True)
    f = extract_features({"e1": regular_train(1.0, 60.0), "e2": np.array([])}, 60.0, p)
    assert f["active_electrodes"] == 1


def test_bursting_denominator_divides_by_electrodes_that_burst() -> None:
    """15 bursts in a minute on 2 of 4 active electrodes: 3.75 per active, 7.5 per bursting."""
    p = FeatureParams(burst_rate_denominator="bursting")
    f = extract_features(_mixed_plate(), 60.0, p)
    assert f["burst_rate_per_min"] == pytest.approx(7.5)
    # Only the rate moves: every other feature is the default's.
    rest = {k: v for k, v in f.items() if k != "burst_rate_per_min"}
    assert rest == {k: v for k, v in GOLDEN_DEFAULTS.items() if k != "burst_rate_per_min"}


def test_bursting_denominator_is_undefined_without_bursts() -> None:
    p = FeatureParams(burst_rate_denominator="bursting")
    f = extract_features({"e1": regular_train(1.0, 60.0)}, 60.0, p)
    assert f["burst_rate_per_min"] is None


def test_unknown_burst_rate_denominator_is_rejected() -> None:
    with pytest.raises(ValueError, match="burst_rate_denominator"):
        FeatureParams(burst_rate_denominator="wells")

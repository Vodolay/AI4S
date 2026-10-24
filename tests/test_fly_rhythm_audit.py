"""Validation of the corrected rhythm measure, as registered for the audit (synthetic signals)."""

from __future__ import annotations

import numpy as np

from neuromps.fly import fictive, kinematics, muscle_map

T = np.arange(0, 1.8, 0.001)


def _pool(rate_hz, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    trains = []
    for _ in range(n):
        spikes = T[rng.random(T.size) < rate_hz * 0.001]
        trains.append(muscle_map.filtered_rate(spikes, T[-1] + 0.001, 0.001, 0.02))
    return np.mean(trains, axis=0)


def test_a_sine_scores_high_with_its_own_period() -> None:
    score, period = kinematics.rhythm(np.sin(2 * np.pi * 8 * T), start_s=0.0)
    assert score >= 0.9 and abs(period - 0.125) <= 0.002


def test_smooth_signals_without_a_rhythm_score_zero() -> None:
    for x in (np.exp(-(((T - 0.15) / 0.06) ** 2)), np.clip((T - 0.2) / 0.15, 0, 1), T**2):
        assert kinematics.rhythm(x, start_s=0.0) == (0.0, None)
        assert kinematics.rhythmicity(x, start_s=0.0) > 0.6  # the old measure's flaw


def test_noise_pools_stay_below_and_modulated_pools_above_the_registered_line() -> None:
    flat = [
        kinematics.rhythm(_pool(np.full(T.size, 30.0), 5, s), start_s=0.0)[0] for s in range(20)
    ]
    beat = 30 * (1 + 0.8 * np.sin(2 * np.pi * 8 * T))
    mod = [kinematics.rhythm(_pool(beat, 5, s), start_s=0.0)[0] for s in range(20)]
    assert np.median(flat) < 0.3 <= np.median(mod)


def test_the_audited_phase_needs_a_peak_in_the_reference() -> None:
    a = np.sin(2 * np.pi * 8 * T)
    b = np.sin(2 * np.pi * 8 * (T - 0.0625))
    assert abs(fictive.phase(a, b, method="audited") - 0.5) <= 0.02
    burst = np.exp(-(((T - 0.15) / 0.06) ** 2))
    assert fictive.phase(burst, b, method="audited") is None
    assert fictive.phase(burst, b) is not None  # the registered measure still returns one


def test_comparison_flags_changed_outcomes_and_reproducibility() -> None:
    from neuromps.fly import rhythm_audit

    got = {
        "registered": {"X": {"value": 0.5, "outcome": "pass"}},
        "audited": {"X": {"value": 0.0, "outcome": "fail"}},
    }
    out = rhythm_audit.compare(
        got, {"predictions": {"X": {"value": 0.5, "outcome": "pass"}}}, ["X"]
    )
    assert out["reproducible"] and out["predictions"]["X"]["changed"]
    out = rhythm_audit.compare(
        got, {"predictions": {"X": {"value": 0.4, "outcome": "pass"}}}, ["X"]
    )
    assert not out["reproducible"]

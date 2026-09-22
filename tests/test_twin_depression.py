"""The depression comparison: the summary that decides whether it helps."""

import numpy as np
import pandas as pd

from neuromps.twin.depression import DEPRESSION_GRID, GAP_FEATURES, summarise
from neuromps.twin.network import NetworkParams, simulate, synapse_counts


def _scored(rows: list[dict]) -> pd.DataFrame:
    """A scored sweep frame in the shape `sweep` returns."""
    return pd.DataFrame(
        [
            {
                "depression_u": u,
                "depression_tau_rec_ms": tau,
                "is_control": u == 0.0,
                "seed": seed,
                "score": score,
                "penalty_synchrony_index": synchrony,
                "penalty_pct_spikes_in_bursts": bursts,
                "value_synchrony_index": 0.5,
            }
            for u, tau, seed, score, synchrony, bursts in rows
        ]
    )


def test_the_grid_contains_a_no_depression_control() -> None:
    assert any(u == 0.0 for u, _ in DEPRESSION_GRID)


def test_summarise_names_the_control_and_the_best_setting() -> None:
    scored = _scored(
        [
            (0.0, 600.0, 0, 0.30, 0.13, 0.03),
            (0.4, 600.0, 0, 0.10, 0.04, 0.01),
        ]
    )
    out = summarise(scored)
    assert out["control"] == "u0_tau600"
    assert out["best"] == "u0.4_tau600"
    assert out["best_beats_control"] is True
    for feature in GAP_FEATURES:
        assert out[f"{feature}_improves_under"] == ["u0.4_tau600"]


def test_summarise_reports_no_improvement_when_depression_makes_it_worse() -> None:
    scored = _scored(
        [
            (0.0, 600.0, 0, 0.10, 0.04, 0.01),
            (0.4, 600.0, 0, 0.30, 0.13, 0.03),
        ]
    )
    out = summarise(scored)
    assert out["best"] == "u0_tau600"
    assert out["best_beats_control"] is False
    for feature in GAP_FEATURES:
        assert out[f"{feature}_improves_under"] == []


def test_summarise_averages_over_seeds_before_comparing() -> None:
    scored = _scored(
        [
            (0.0, 600.0, 0, 0.20, 0.10, 0.02),
            (0.0, 600.0, 1, 0.40, 0.20, 0.04),
            (0.4, 600.0, 0, 0.25, 0.12, 0.02),
            (0.4, 600.0, 1, 0.29, 0.14, 0.03),
        ]
    )
    out = summarise(scored)
    # Control averages 0.30 and the setting 0.27, though the control wins on
    # one of the two seeds.
    assert out["settings"]["u0_tau600"]["score"] == 0.30
    assert out["settings"]["u0.4_tau600"]["score"] == 0.27
    assert out["best_beats_control"] is True


def test_summarise_says_so_rather_than_guessing_when_nothing_ran() -> None:
    assert "status" in summarise(pd.DataFrame())
    empty = _scored([(0.0, 600.0, 0, None, None, None)])
    assert "status" in summarise(empty)


def test_depression_does_not_change_the_wiring() -> None:
    """The comparison is only fair if the two runs are the same network."""
    plain = NetworkParams(n_per_compartment=60, duration_s=1.0, seed=2)
    depressing = NetworkParams(n_per_compartment=60, duration_s=1.0, seed=2, depression_u=0.5)
    assert synapse_counts(plain) == synapse_counts(depressing)


def test_the_default_is_exactly_the_model_without_depression() -> None:
    """Every result produced before this parameter existed must reproduce."""
    chip = dict(n_per_compartment=80, duration_s=3.0, seed=4)
    plain = simulate(NetworkParams(**chip))
    same = simulate(NetworkParams(**chip, depression_u=0.0))
    assert np.array_equal(plain.spike_times, same.spike_times)


def test_depression_changes_the_dynamics_when_switched_on() -> None:
    """That it is wired in at all - which way it moves the features is §7.3's sweep.

    Not a directional claim: depression also weakens the excitation driving the
    inhibitory cells, so the net effect on total rate is not signed a priori.
    That is exactly why `twin/depression.py` measures it instead of assuming it.
    """
    chip = dict(n_per_compartment=80, duration_s=3.0, seed=4)
    plain = simulate(NetworkParams(**chip))
    strong = simulate(NetworkParams(**chip, depression_u=0.8, depression_tau_rec_ms=900.0))
    assert not np.array_equal(plain.spike_times, strong.spike_times)

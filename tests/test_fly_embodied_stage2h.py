"""Stage 2h: the bisection step and the renaming of stage 2g's scores."""

from __future__ import annotations

import json

import numpy as np

from neuromps.fly import embodied_stage2h as s2h


def test_bracketed_draws_take_the_geometric_mean_others_halve_or_double() -> None:
    inputs = np.array([100.0, 100.0, 100.0])
    recruited = np.array([900, 2, 900])
    lo = np.array([25.0, np.nan, np.nan])
    hi = np.array([100.0, np.nan, 100.0])
    np.testing.assert_allclose(s2h.next_input(inputs, recruited, lo, hi), [50.0, 200.0, 50.0])


def test_scores_carry_this_registrations_names_and_tests() -> None:
    import pandas as pd

    spec = json.loads(s2h.REGISTRATION.read_text())
    table = pd.DataFrame(
        [
            {
                "condition": c,
                "feasible": True,
                "score": 0.6 if c == "forward_walking" else 0.05,
                "phase_promotor_remotor": 0.3,
                "phase_flexor_extensor": 0.5,
                "period_ms": 100.0,
            }
            for c in ("forward_walking", "grooming", "lesion")
            for _ in range(32)
        ]
    )
    got = s2h.score(table, spec)
    assert list(got) == list(spec["predictions"])
    assert all(v["outcome"] == "pass" for v in got.values())
    assert (
        got["H0_a_working_input_exists"]["test"]
        == spec["predictions"]["H0_a_working_input_exists"]["test"]
    )

"""Stage 2g: the front-leg network construction, the recruitment rule, the scoring rules."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from neuromps.fly import connectome
from neuromps.fly import embodied_stage2g as s2g


def _net() -> connectome.Connectome:
    """0 = MN; 1 premotor (6 syn onto 0); 2 weak (3 syn onto 0); 3 DN onto 1; 4 DN onto 2;
    5 DN onto 0 only; 6 unrelated."""
    return connectome.Connectome(
        root_ids=np.arange(7),
        pre=np.array([1, 2, 3, 4, 5, 6]),
        post=np.array([0, 0, 1, 2, 0, 6]),
        synapses=np.array([6, 3, 9, 9, 9, 9]),
        sign=np.ones(6, int),
        pre_class=np.array(["ach"] * 6, dtype=object),
        neuropil=np.array(["A"] * 7, dtype=object),
    )


def test_the_network_keeps_motor_premotor_and_descending_onto_premotor() -> None:
    keep = s2g.front_leg_network(_net(), motor=np.array([0]), descending=np.array([3, 4, 5]))
    np.testing.assert_array_equal(np.flatnonzero(keep), [0, 1, 3])


def test_the_recruitment_rule_halves_doubles_or_keeps() -> None:
    got = s2g.next_inputs(np.array([250.0, 250.0, 250.0]), np.array([900, 2, 40]))
    np.testing.assert_allclose(got, [125.0, 500.0, 250.0])


def _table(score_fw: float, feasible: int = 32) -> pd.DataFrame:
    rows = []
    for cond in ("forward_walking", "grooming", "lesion"):
        for k in range(32):
            rows.append(
                {
                    "condition": cond,
                    "feasible": k < feasible,
                    "score": score_fw if cond == "forward_walking" else 0.05,
                    "phase_promotor_remotor": 0.3,
                    "phase_flexor_extensor": 0.5,
                    "period_ms": 100.0,
                }
            )
    return pd.DataFrame(rows)


SPEC = json.loads(s2g.REGISTRATION.read_text())


def test_a_reliable_rhythm_passes_everything() -> None:
    got = s2g.score(_table(0.6), SPEC)
    assert all(v["outcome"] == "pass" for v in got.values()), got


def test_too_few_feasible_draws_make_the_rest_uninterpretable() -> None:
    got = s2g.score(_table(0.6, feasible=10), SPEC)
    assert got["P0_the_rule_finds_a_working_input"]["outcome"] == "fail"
    assert {v["outcome"] for k, v in got.items() if not k.startswith("P0")} == {"uninterpretable"}


def test_no_rhythm_means_no_lesion_verdict() -> None:
    got = s2g.score(_table(0.1), SPEC)
    assert got["P5_the_rhythm_needs_the_three_neuron_circuit"]["outcome"] == "uninterpretable"


def test_scattered_phases_are_not_reliable() -> None:
    phases = pd.Series(np.linspace(0, 1, 20, endpoint=False))
    assert s2g.resultant(phases) < 0.1
    assert s2g.resultant(pd.Series([0.3] * 5)) > 0.99

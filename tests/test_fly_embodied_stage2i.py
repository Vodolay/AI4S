"""Stage 2i: fresh draws, the volume rule's name, and the renaming of stage 2g's scores."""

from __future__ import annotations

import json

import pandas as pd

from neuromps.fly import embodied_stage2h as s2h
from neuromps.fly import embodied_stage2i as s2i


def test_fresh_draws_and_the_volume_rule() -> None:
    assert len(s2i.SEEDS) == 32 and not set(s2i.SEEDS) & set(s2h.SEEDS)
    assert s2i.RULE == "volume"


def test_scores_carry_this_registrations_names_and_tests() -> None:
    spec = json.loads(s2i.REGISTRATION.read_text())
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
    got = s2i.score(table, spec)
    assert list(got) == list(spec["predictions"])
    assert all(v["outcome"] == "pass" for v in got.values())
    name = "V5_the_rhythm_needs_the_three_neuron_circuit"
    assert got[name]["test"] == spec["predictions"][name]["test"]
